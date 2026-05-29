import os
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .models import VocalSumPipeline, Transcriber, PEGASUSSummarizer
from .audio import AudioSeparator

app = FastAPI(
    title="VocalSum API",
    description="REST API for audio vocal separation, ASR transcription, translation, and abstractive summarization.",
    version="0.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load global models on startup
# We keep paths configurable or fallback to defaults
unet_path = os.getenv("UNET_CHECKPOINT", "checkpoints/unet.pt")
pegasus_path = os.getenv("PEGASUS_CHECKPOINT", "google/pegasus-cnn_dailymail")
whisper_size = os.getenv("WHISPER_MODEL", "medium")

# Lazily initialized pipeline
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        # If weights do not exist yet, they will run fallback/untrained configurations
        pipeline = VocalSumPipeline(
            unet_path=unet_path if os.path.exists(unet_path) else None,
            whisper_size=whisper_size,
            pegasus_path=pegasus_path,
        )
    return pipeline

# Temp file directory
TEMP_DIR = "temp_api_files"
os.makedirs(TEMP_DIR, exist_ok=True)

class SummarizeRequest(BaseModel):
    text: str
    num_beams: Optional[int] = 8
    max_length: Optional[int] = 128
    length_penalty: Optional[float] = 0.8

class SummaryResponse(BaseModel):
    summary: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "VocalSum"}

@app.post("/api/separate")
async def separate_audio(file: UploadFile = File(...)):
    """
    Stage 1: Separates vocals and accompaniment from uploaded mixed audio.
    """
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        sep = AudioSeparator(model_path=unet_path if os.path.exists(unet_path) else None)
        out_dir = os.path.join(TEMP_DIR, "output_" + os.path.splitext(file.filename)[0])
        sep.separate(file_path, output_dir=out_dir)
        
        return {
            "vocals_url": f"/api/files/download?path={os.path.join(out_dir, 'vocals.wav')}",
            "accompaniment_url": f"/api/files/download?path={os.path.join(out_dir, 'accompaniment.wav')}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    """
    Stage 2: Transcribes vocal track using OpenAI Whisper ASR.
    """
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        transcriber = Transcriber(model_size=whisper_size)
        result = transcriber.transcribe(file_path, language=language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/summarize", response_model=SummaryResponse)
async def summarize_text(req: SummarizeRequest):
    """
    Stage 3: Generates an abstractive summary using PEGASUS.
    """
    try:
        summarizer = PEGASUSSummarizer(model_name=pegasus_path)
        summary = summarizer.summarize(
            req.text,
            num_beams=req.num_beams,
            max_length=req.max_length,
            length_penalty=req.length_penalty
        )
        return SummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process")
async def run_full_pipeline(
    file: UploadFile = File(...), 
    language: Optional[str] = Form(None)
):
    """
    End-to-end processing pipeline runs vocal separation, speech recognition,
    Hindi-English translation (if needed), and abstractive summarization.
    """
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        pipe = get_pipeline()
        out_dir = os.path.join(TEMP_DIR, "output_" + os.path.splitext(file.filename)[0])
        result = pipe.process(file_path, target_language=language, output_dir=out_dir)
        
        result["vocals_url"] = f"/api/files/download?path={result.get('vocal_path', '')}"
        result["accompaniment_url"] = f"/api/files/download?path={result.get('accomp_path', '')}"
        
        # Clean paths inside result to prevent internal server details leak
        result.pop("vocal_path", None)
        result.pop("accomp_path", None)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/api/files/download")
def download_file(path: str):
    """
    Helper endpoint to download outputs.
    """
    # Simple sanitization
    clean_path = os.path.normpath(path)
    if not clean_path.startswith(TEMP_DIR):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.exists(clean_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(clean_path, media_type="audio/wav")
