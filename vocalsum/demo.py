import os
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import librosa
from .models import VocalSumPipeline
from .audio import compute_stft

# Load the pipeline lazily
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        unet_path = os.getenv("UNET_CHECKPOINT", "checkpoints/unet.pt")
        pegasus_path = os.getenv("PEGASUS_CHECKPOINT", "google/pegasus-cnn_dailymail")
        whisper_size = os.getenv("WHISPER_MODEL", "medium")
        pipeline = VocalSumPipeline(
            unet_path=unet_path if os.path.exists(unet_path) else None,
            whisper_size=whisper_size,
            pegasus_path=pegasus_path,
        )
    return pipeline

def create_spectrogram_plot(audio_path, title):
    """
    Helper to compute and plot spectrogram for visualizations.
    Returns path of saved plot image.
    """
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        magnitude, _ = compute_stft(y)
        
        # Plot
        plt.figure(figsize=(6, 3))
        librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max), 
                                 y_axis='log', x_axis='time', sr=sr, cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        
        plot_path = f"temp_{title.lower().replace(' ', '_')}.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    except Exception as e:
        print("Spectrogram plotting error:", e)
        return None


def run_pipeline(audio_file, language_choice):
    """
    Executes VocalSum pipeline and yields outputs.
    """
    if audio_file is None:
        return None, None, None, None, None, None, "Please upload an audio file."
        
    try:
        pipe = get_pipeline()
        
        # Create temp folder for outputs
        output_dir = "temp_demo_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        lang = None if language_choice == "Auto-Detect" else ("hi" if language_choice == "Hindi" else "en")
        
        # Process audio
        result = pipe.process(audio_file, target_language=lang, output_dir=output_dir)
        
        vocal_path = result.get("vocal_path", os.path.join(output_dir, "vocals.wav"))
        accomp_path = result.get("accomp_path", os.path.join(output_dir, "accompaniment.wav"))
        
        # Generate Spectrogram Plots
        mix_spec = create_spectrogram_plot(audio_file, "Original Mixed Spectrogram")
        vocal_spec = create_spectrogram_plot(vocal_path, "Clean Vocals Spectrogram")
        accomp_spec = create_spectrogram_plot(accomp_path, "Accompaniment Spectrogram")
        
        translation_text = result.get("translated_transcript") or "N/A (English input)"
        
        return (
            vocal_path, 
            accomp_path, 
            mix_spec,
            vocal_spec,
            accomp_spec,
            result.get("transcript", ""), 
            translation_text, 
            result.get("summary", "")
        )
    except Exception as e:
        return None, None, None, None, None, f"Error occurred: {str(e)}", "", ""


# Custom Premium Theme
custom_css = """
body {
    background: #0f172a;
    color: #f1f5f9;
}
.gradio-container {
    font-family: 'Outfit', 'Inter', sans-serif !important;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.95);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}
"""

with gr.Blocks(css=custom_css, title="VocalSum — Vocal Separation, Transcription & Summarization") as demo:
    gr.HTML("<div style='text-align: center; margin-bottom: 20px;'><h1 style='color: #38bdf8; font-weight: 800;'>🎙️ VocalSum Pipeline</h1><p style='color: #94a3b8;'>Separate vocals, transcribe speech (multilingual), and perform abstractive summarization in one single step.</p></div>")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(label="Upload Mixed Audio File", type="filepath")
            language_dropdown = gr.Dropdown(choices=["Auto-Detect", "English", "Hindi"], value="Auto-Detect", label="ASR Vocal Language Hint")
            submit_btn = gr.Button("Process Audio", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("🎵 Audio Stems"):
                    vocal_audio = gr.Audio(label="Clean Vocals (Stage 1)", type="filepath")
                    accomp_audio = gr.Audio(label="Background Accompaniment (Stage 1)", type="filepath")
                    
                with gr.TabItem("📊 Spectrogram Analysis"):
                    with gr.Row():
                        mix_plot = gr.Image(label="Original Mix Spectrogram")
                        vocal_plot = gr.Image(label="Vocals Spectrogram")
                        accomp_plot = gr.Image(label="Accompaniment Spectrogram")
                        
                with gr.TabItem("📝 Text Analysis"):
                    transcript_box = gr.Textbox(label="Vocal Transcript (Stage 2)", lines=5)
                    translation_box = gr.Textbox(label="Hindi-to-English Translation (Optional)", lines=5)
                    summary_box = gr.Textbox(label="Abstractive Summary (Stage 3)", lines=5, placeholder="Final PEGASUS generated summary...")

    submit_btn.click(
        fn=run_pipeline,
        inputs=[audio_input, language_dropdown],
        outputs=[
            vocal_audio, 
            accomp_audio, 
            mix_plot,
            vocal_plot,
            accomp_plot,
            transcript_box, 
            translation_box, 
            summary_box
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
