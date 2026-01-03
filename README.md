# 🚨 Emergency Calls Priority System

An AI-powered system for analyzing 911 emergency call recordings. It automatically transcribes audio, identifies speakers, and assigns urgency ratings to help dispatchers prioritize emergency responses effectively.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern%20API-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎙️ **Speech-to-Text** | Converts audio recordings to text using OpenAI Whisper |
| 👥 **Speaker Diarization** | Identifies and labels different speakers (Caller/Operator) |
| 🔥 **Urgency Rating** | AI-powered 0-5 urgency scale assessment |
| 📊 **Dashboard** | Real-time monitoring and management interface |
| 📁 **Multi-Format Export** | Download transcripts in JSON, TXT, or Summary formats |
| ⚡ **Queue Processing** | Handles multiple uploads with background processing |

---

## 🎯 Urgency Rating Scale

| Rating | Level | Description |
|--------|-------|-------------|
| 0 | No Emergency | Routine or informational conversation |
| 1 | Minor Concern | Low-risk situation, no immediate action needed |
| 2 | Moderate Concern | Some urgency, assistance may be needed soon |
| 3 | High Concern | Urgent situation requiring prompt attention |
| 4 | Critical Emergency | Serious emergency, quick response essential |
| 5 | Life-Threatening | Extreme urgency, immediate action required |

---

## 🛠️ Tech Stack

- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Database:** SQLite
- **AI Models:** Whisper (STT), Pyannote (Diarization), Gemma 3 (LLM)
- **ML Framework:** PyTorch, Transformers, faster-whisper

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended)
- [Hugging Face Account](https://huggingface.co/) with access token
- Minimum 8GB RAM (16GB recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/sujitsoni3804/Emergency-Calls-Priority-System.git
cd Emergency-Calls-Priority-System
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Environment Variables

```bash
# Windows (PowerShell)
$env:HF_TOKEN = "your_huggingface_token"

# Linux/Mac
export HF_TOKEN="your_huggingface_token"
```

> **Note:** Get your Hugging Face token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Step 5: Download AI Models

Download the Gemma 3 4B model and place it in the `models/gemma-3-4b-it/` directory.

### Step 6: Run the Application

```bash
python app.py
```

The application will be available at:
- **Local:** http://127.0.0.1:5000
- **Network:** http://YOUR_IP:5000

---

## 🎮 Usage

1. **Upload Audio:** Navigate to the homepage and upload 911 call recordings (supports common audio formats)
2. **Processing:** The system automatically transcribes, diarizes, and analyzes the audio
3. **View Results:** Access transcripts with speaker labels and urgency ratings
4. **Dashboard:** Monitor all processed calls, sorted by urgency level
5. **Export:** Download transcripts in your preferred format

---

## 🤖 AI Models Used

| Model | Purpose | URL |
|-------|---------|-----|
| **Whisper Small EN** | Speech-to-Text | [openai/whisper-small.en](https://huggingface.co/openai/whisper-small.en) |
| **Pyannote Diarization 3.1** | Speaker Identification | [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) |
| **Gemma 3 4B** | Summarization & Urgency Rating | [gemma3:4b](https://ollama.com/library/gemma3:4b) |

---

## 📊 Datasets

| Source | Description | URL |
|--------|-------------|-----|
| **Kaggle** | 911 Emergency Calls Dataset (708 recordings) | [911 Recordings Dataset](https://www.kaggle.com/datasets/louisteitelbaum/911-recordings) |
| **YouTube** | Playlist of 911 Emergency Call recordings | [Emergency Calls Playlist](https://youtube.com/playlist?list=PLplPIyOskVhxAgGQsNjRoXykxV0u41zbl&si=GjYaEvTJM1rmRUBs) |

---

## 📁 Project Structure

```
Emergency-Calls-Priority-System/
├── app.py                 # FastAPI application entry point
├── requirements.txt       # Python dependencies
├── transcriptions.db      # SQLite database
├── scripts/
│   ├── audio_processing.py   # Audio processing pipeline
│   ├── config.py             # Configuration settings
│   ├── database.py           # Database operations
│   ├── diarization.py        # Speaker diarization logic
│   ├── file_handler.py       # File management utilities
│   ├── models.py             # AI model loading & inference
│   ├── routes.py             # API endpoints
│   └── summarization.py      # LLM summarization & urgency rating
├── templates/
│   ├── index.html        # Upload interface
│   ├── dashboard.html    # Monitoring dashboard
│   └── view.html         # Transcript viewer
├── uploads/              # Uploaded audio files
├── outputs/              # Generated transcripts
└── models/               # AI model weights
```

---

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [Google Gemma](https://ai.google.dev/gemma) for language understanding

---

<p align="center">
  Made with ❤️ for Emergency Response Optimization
</p>
