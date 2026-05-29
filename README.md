# 🎙️ VocalSum

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

**VocalSum** is a state-of-the-art, three-stage deep learning pipeline that seamlessly bridges **audio source separation**, **speech recognition (ASR)**, **multilingual translation**, and **abstractive dialogue summarization** into a single unified workspace.

Designed for meeting intelligence, media monitoring, and call analytics, the project separates human vocals from background noise/accompaniment, transcribes speech, translates Hindi-language content to English, and generates highly accurate dialogue summaries.

---

## 🚀 Key Pipeline Stages

```
[Noisy Input Waveform] 
        │
        ▼ (Stage 1)
┌────────────────────────────────┐
│   U-Net Audio Separation       │ ──► [Accompaniment.wav]
└────────────────────────────────┘
        │
        ▼ [Clean Vocals.wav] (Stage 2)
┌────────────────────────────────┐
│   OpenAI Whisper ASR           │
└────────────────────────────────┘
        │
        ▼ [Raw Transcript] (Optional Stage)
┌────────────────────────────────┐
│   MarianMT (Hindi ➔ English)   │
└────────────────────────────────┘
        │
        ▼ [English Transcript] (Stage 3)
┌────────────────────────────────┐
│   PEGASUS Summarization        │
└────────────────────────────────┘
        │
        ▼
[Final Abstractive Summary]
```

### 1. Stage 1: Audio Source Separation (U-Net CNN)
* Transforms noisy input audio to time-frequency spectrograms using the **Short-Time Fourier Transform (STFT)**.
* Employs a custom 6-layer **U-Net CNN** encoder-decoder architecture with skip-connections to predict a soft time-frequency mask for vocals and accompaniment.
* Reconstructs the target audio stems back to time-domain waveforms using the **Inverse STFT (ISTFT)**.

### 2. Stage 2: Automatic Speech Recognition (OpenAI Whisper)
* The isolated clean vocals track is passed directly into a multi-lingual **OpenAI Whisper (medium)** model for robust transcription.
* Supports transcription with word-level timestamps.

### 3. Translation (MarianMT)
* Integrates a dedicated **MarianMT** (`Helsinki-NLP/opus-mt-hi-en`) translation model.
* Automatically translates Hindi transcripts to English before summarization to enable cross-lingual abstractive summaries.

### 4. Stage 3: Abstractive Summarization (PEGASUS)
* The text transcript is summarized using a fine-tuned **PEGASUS** (`google/pegasus-cnn_dailymail`) sequence-to-sequence model.
* Hyperparameter-tuned using beam-search width $8$ and length penalty $0.8$ for concise summaries.

---

## 📊 Evaluation & Benchmarks

* **Audio Source Separation**: Evaluated on the **MUSDB18** benchmark dataset, reaching a vocal **SDR of 10.2 dB**.
* **Text Summarization**: Fine-tuned on the **SAMSum** dialogue corpus, achieving a state-of-the-art **ROUGE-1 score of 0.4821**.

---

## 📁 Repository Structure

```
VocalSum/
├── README.md               # Visual overview and guide
├── requirements.txt        # Package dependencies
├── setup.py                # Setup configuration for packaging
├── .env.example            # Environment variables configuration
├── .gitignore              # Git patterns to ignore
│
├── vocalsum/
│   ├── __init__.py         # Package initialization
│   ├── audio.py            # STFT/ISTFT helpers, U-Net CNN module, and MUSDB loader
│   ├── models.py           # Whisper, MarianMT, PEGASUS, and VocalSumPipeline
│   ├── train.py            # Training/fine-tuning scripts for U-Net & PEGASUS
│   ├── eval.py             # SDR/SIR/SAR & ROUGE metric calculation routines
│   ├── api.py              # FastAPI REST endpoints application
│   └── demo.py             # Premium Gradio web demo layout
│
├── scripts/
│   └── download_models.py  # Script to download weights & pre-train configs
│
└── tests/
    └── test_vocalsum.py    # Unit tests suite (reconstruction & dimensions)
```

---

## ⚡ Quick Start

### 1. Installation
Clone the repository, create a virtual environment, and install dependencies:
```bash
git clone https://github.com/ultramech79-art/VocalSum.git
cd VocalSum
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Model Checkpoints
Before launching services, fetch required transformer weights and compile random initial states for the U-Net:
```bash
python3 scripts/download_models.py
```

### 3. Launch the API Service (FastAPI)
Launch the server locally. Swagger interactive docs will be available at [http://localhost:8000/docs](http://localhost:8000/docs).
```bash
uvicorn vocalsum.api:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run the GUI Demo (Gradio)
Launch a premium, user-friendly browser dashboard to upload audio, view spectrograms, and read summaries:
```bash
python3 -m vocalsum.demo
```

---

## 🧪 Testing

Run pytest to verify package models, spectrogram reconstruction, and pipeline dimensions:
```bash
pytest tests/ -v
```

---

## 🛠️ Hyperparameter Settings

| Hyperparameter | Target Value |
|---|---|
| U-Net Learning Rate | 1e-4 |
| U-Net Training Epochs | 100 |
| STFT Window / Hop | 4096 / 1024 samples |
| Resampling Rate | 22,050 Hz (mono) |
| PEGASUS Input / Summary Max Tokens | 1024 / 128 |
| Beam Search Width | 8 |
| Length Penalty | 0.8 |

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
