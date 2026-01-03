import os
import uuid
import shutil
import json
import asyncio
import threading
import queue
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from typing import List
from scripts.config import UPLOAD_FOLDER, OUTPUT_FOLDER, TEMPLATE_FOLDER, processing_status
from scripts.models import init_models
from scripts.audio_processing import process_audio
from scripts.database import add_transcription, update_status, get_all_transcriptions, get_transcription

router = APIRouter()

processing_queue = queue.Queue()
queue_lock = threading.Lock()
is_processing = False

def queue_worker():
    global is_processing
    while True:
        try:
            job = processing_queue.get(timeout=1)
            if job is None:
                break
            with queue_lock:
                is_processing = True
            file_path, job_id = job
            process_audio(file_path, job_id)
            with queue_lock:
                is_processing = False
            processing_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Queue worker error: {e}")
            with queue_lock:
                is_processing = False

worker_thread = threading.Thread(target=queue_worker, daemon=True)
worker_thread.start()
 
@router.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(TEMPLATE_FOLDER, "index.html")
    with open(html_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html_path = os.path.join(TEMPLATE_FOLDER, "dashboard.html")
    with open(html_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@router.get("/view/{job_id}", response_class=HTMLResponse)
async def view_transcript(job_id: str):
    html_path = os.path.join(TEMPLATE_FOLDER, "view.html")
    with open(html_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    init_models()
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    job_id = str(uuid.uuid4())
    filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
    name_parts = os.path.splitext(filename)
    new_filename = f"{name_parts[0]}_{job_id}{name_parts[1]}"
    file_path = os.path.join(UPLOAD_FOLDER, new_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    add_transcription(job_id, filename)
    processing_queue.put((file_path, job_id))
    return {"job_id": job_id, "filename": filename}

@router.post("/upload-multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    init_models()
    if not files:
        raise HTTPException(status_code=400, detail="No files selected")
    job_ids = []
    for file in files:
        if not file.filename:
            continue
        job_id = str(uuid.uuid4())
        filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
        name_parts = os.path.splitext(filename)
        new_filename = f"{name_parts[0]}_{job_id}{name_parts[1]}"
        file_path = os.path.join(UPLOAD_FOLDER, new_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        add_transcription(job_id, filename)
        processing_queue.put((file_path, job_id))
        job_ids.append({"job_id": job_id, "filename": filename})
    return {"jobs": job_ids}

@router.get("/progress/{job_id}")
async def progress(job_id: str):
    async def event_generator():
        while True:
            if job_id in processing_status:
                status = processing_status[job_id]
                yield f"data: {json.dumps(status)}\n\n"
                if status["progress"] == 100 or status["progress"] == 0:
                    break
            await asyncio.sleep(0.5)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/summary_status/{job_id}")
async def summary_status(job_id: str):
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    status = processing_status[job_id]
    return {
        "summary_status": status.get("summary_status", "pending"),
        "summary": status.get("summary", "")
    }

@router.get("/audio/{job_id}")
async def get_audio(job_id: str):
    for filename in os.listdir(UPLOAD_FOLDER):
        if f"_{job_id}." in filename or filename.endswith(f"_{job_id}"):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Audio not found")

@router.get("/api/transcript/{job_id}")
async def get_transcript_data(job_id: str):
    json_file = None
    for filename in os.listdir(OUTPUT_FOLDER):
        if f"_{job_id}.json" in filename:
            json_file = filename
            break
    
    if not json_file:
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    json_path = os.path.join(OUTPUT_FOLDER, json_file)
    with open(json_path, 'r', encoding='utf-8') as f:
        llm_format_data = json.load(f)
    
    transcript = []
    speaker_map = {}
    speaker_counter = 0
    
    for idx, item in enumerate(llm_format_data):
        for speaker_label, text in item.items():
            if speaker_label not in speaker_map.values():
                speaker_key = f"SPEAKER_{speaker_counter:02d}"
                speaker_map[speaker_key] = speaker_label
                speaker_counter += 1
            
            speaker_key = [k for k, v in speaker_map.items() if v == speaker_label][0]
            
            transcript.append({
                "speaker": speaker_key,
                "text": text,
                "start": 0,
                "end": 0
            })
    
    db_data = get_transcription(job_id)
    
    original_filename = "Unknown"
    if db_data and "filename" in db_data:
        original_filename = db_data["filename"]
    
    response_data = {
        "job_id": job_id,
        "filename": original_filename,
        "transcript": transcript,
        "speaker_map": speaker_map,
        "time_taken": 0,
        "summary": db_data.get("summary", "") if db_data else ""
    }
    
    return response_data

@router.get("/download/{job_id}/{format}")
async def download_transcript(job_id: str, format: str):
    for filename in os.listdir(OUTPUT_FOLDER):
        if f"_{job_id}" in filename:
            if format == 'json' and filename.endswith('.json'):
                file_path = os.path.join(OUTPUT_FOLDER, filename)
                download_name = filename.rsplit(f"_{job_id}", 1)[0] + ".json"
                return FileResponse(file_path, filename=download_name, media_type='application/json')
            elif format == 'txt' and filename.endswith('.txt') and '_summary' not in filename:
                file_path = os.path.join(OUTPUT_FOLDER, filename)
                download_name = filename.rsplit(f"_{job_id}", 1)[0] + ".txt"
                return FileResponse(file_path, filename=download_name, media_type='text/plain')
            elif format == 'summary' and filename.endswith('_summary.txt'):
                file_path = os.path.join(OUTPUT_FOLDER, filename)
                download_name = filename.rsplit(f"_{job_id}", 1)[0] + "_summary.txt"
                return FileResponse(file_path, filename=download_name, media_type='text/plain')
    raise HTTPException(status_code=404, detail="File not found")

@router.get("/api/transcriptions")
async def get_transcriptions():
    return {"transcriptions": get_all_transcriptions()}

@router.post("/api/transcriptions/{job_id}/status")
async def update_transcription_status(job_id: str, status: dict):
    new_status = status.get("status")
    if new_status not in ["pending", "completed"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    update_status(job_id, new_status)
    return {"success": True}

@router.delete("/api/transcriptions/{job_id}")
async def delete_transcription(job_id: str):
    from scripts.database import delete_transcription
    
    deleted_files = []
    
    for filename in os.listdir(UPLOAD_FOLDER):
        if f"_{job_id}" in filename:
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            os.remove(file_path)
            deleted_files.append(filename)
    
    for filename in os.listdir(OUTPUT_FOLDER):
        if f"_{job_id}" in filename:
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            os.remove(file_path)
            deleted_files.append(filename)
    
    delete_transcription(job_id)
    
    return {"success": True, "deleted_files": deleted_files}