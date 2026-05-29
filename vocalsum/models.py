import os
import torch
import whisper
from transformers import MarianMTModel, MarianTokenizer, PegasusForConditionalGeneration, PegasusTokenizer

class Transcriber:
    """
    ASR transcription stage utilizing OpenAI Whisper.
    """
    def __init__(self, model_size="medium", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load whisper model
        self.model = whisper.load_model(model_size, device=self.device)

    def transcribe(self, audio, language=None):
        """
        Transcribes vocal audio (either a path string or numpy waveform array).
        Returns: A dict with "text", "language", and "segments".
        """
        # Whisper can take a numpy array or a path
        result = self.model.transcribe(audio, language=language, temperature=0.0)
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", "en"),
            "segments": result.get("segments", [])
        }


class HindiEnglishTranslator:
    """
    Translation stage from Hindi to English using Helsinki-NLP/opus-mt-hi-en.
    """
    def __init__(self, model_name="Helsinki-NLP/opus-mt-hi-en", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)

    def translate(self, text):
        """
        Translates Hindi text to English.
        """
        if not text.strip():
            return ""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            translated_tokens = self.model.generate(**inputs)
        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translated_text[0] if translated_text else ""


class PEGASUSSummarizer:
    """
    Abstractive summarization stage utilizing PEGASUS.
    """
    def __init__(self, model_name="google/pegasus-cnn_dailymail", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def summarize(self, text, num_beams=8, max_length=128, length_penalty=0.8):
        """
        Summarizes the provided text transcript.
        """
        if not text.strip():
            return ""
        # Format input and tokenize
        inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                num_beams=num_beams,
                max_length=max_length,
                length_penalty=length_penalty,
                early_stopping=True
            )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


    """
    Consolidated end-to-end processing pipeline orchestrating separation, transcription,
    translation (if needed), and abstractive summarization.
    """
    def __init__(self, unet_path=None, whisper_size="medium", pegasus_path=None, device=None):
        from .audio import AudioSeparator
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instantiate stages
        self.separator = AudioSeparator(model_path=unet_path, device=self.device)
        self.transcriber = Transcriber(model_size=whisper_size, device=self.device)
        self.translator = HindiEnglishTranslator(device=self.device)
        
        pegasus_model = pegasus_path or "google/pegasus-cnn_dailymail"
        self.summarizer = PEGASUSSummarizer(model_name=pegasus_model, device=self.device)

    def process(self, audio_path, target_language=None, output_dir=None):
        """
        Runs the end-to-end process on raw noisy audio.
        Returns a dict containing:
          - transcript: The transcribed text
          - language: Detected ASR language
          - translated_transcript: English transcript (if original was Hindi)
          - summary: The final summary
          - vocal_path/accomp_path: Saved stems if output_dir provided
        """
        import soundfile as sf
        
        # Stage 1: Audio separation
        vocal_wave, accomp_wave, sr = self.separator.separate(audio_path)
        
        # Save temp files for transcription if needed, or transcribe directly from waveform
        # Whisper python library can transcribe float32 numpy array directly
        # but expects it at 16000Hz. Let's write vocals to a temp path or resample.
        temp_vocal_path = "temp_vocals.wav"
        sf.write(temp_vocal_path, vocal_wave, sr)
        
        try:
            # Stage 2: ASR
            asr_result = self.transcriber.transcribe(temp_vocal_path, language=target_language)
            transcript = asr_result["text"]
            detected_lang = asr_result["language"]
            
            # Stage 3: Hindi-English translation (if Hindi)
            translated_transcript = None
            text_for_summary = transcript
            
            if detected_lang == "hi":
                translated_transcript = self.translator.translate(transcript)
                text_for_summary = translated_transcript
                
            # Stage 4: Abstractive Summarization
            summary = self.summarizer.summarize(text_for_summary)
            
            result = {
                "transcript": transcript,
                "language": detected_lang,
                "translated_transcript": translated_transcript,
                "summary": summary
            }
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                vocal_path = os.path.join(output_dir, "vocals.wav")
                accomp_path = os.path.join(output_dir, "accompaniment.wav")
                sf.write(vocal_path, vocal_wave, sr)
                sf.write(accomp_path, accomp_wave, sr)
                result["vocal_path"] = vocal_path
                result["accomp_path"] = accomp_path
                
            return result
            
        finally:
            if os.path.exists(temp_vocal_path):
                os.remove(temp_vocal_path)