# VocalSum

**VocalSum** is an end‑to‑end deep‑learning pipeline that separates vocals from a mixed audio track, transcribes the isolated vocals, translates Hindi to English when needed, and finally generates a concise abstractive summary.

The repository contains:
- Model implementations (U‑Net source‑separation, Whisper ASR, PEGASUS summarizer)
- Training scripts and hyper‑parameter configs
- Evaluation harness for audio‑separation (SDR/SIR/SAR) and summarization (ROUGE)
- A FastAPI REST service and optional Gradio demo UI
- Docker configuration for reproducible GPU‑accelerated deployment

> **Quick start** (GPU machine)
> ```bash
> git clone <repo-url>
> cd VocalSum
> pip install -r requirements.txt
> python scripts/download_models.py   # download pretrained checkpoints
> uvicorn api.app:app --reload
> ```
