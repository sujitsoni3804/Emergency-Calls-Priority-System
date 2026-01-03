import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MAX_CONTENT_LENGTH = 500 * 1024 * 1024
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMPLATE_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("Set HF_TOKEN environment variable")

WHISPER_SIZE = "small.en"
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "int8_float16" if WHISPER_DEVICE == "cuda" else "int8"

processing_status = {}