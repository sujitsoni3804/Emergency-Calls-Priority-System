import os
import tempfile
import time
import torch
import soundfile as sf
import threading
from scripts.config import processing_status
from scripts.models import load_pipeline, unload_pipeline, load_asr, unload_asr, load_llm, unload_llm
from scripts.file_handler import convert_to_audio, save_outputs
from scripts.diarization import perform_diarization, merge_whisper_and_diarization
from scripts.summarization import generate_summary, extract_urgency_rating, save_summary
from scripts.database import update_summary, update_duration

def generate_summary_async(job_id, merged_transcript, speaker_map, base_filename):
    try:
        processing_status[job_id]["summary_status"] = "generating"
        llm_tokenizer, llm_model = load_llm()
        summary = generate_summary(llm_tokenizer, llm_model, merged_transcript, speaker_map)
        urgency_rating = extract_urgency_rating(summary)
        save_summary(job_id, base_filename, summary)
        processing_status[job_id]["summary"] = summary
        processing_status[job_id]["summary_status"] = "completed"
        update_summary(job_id, summary, urgency_rating)
    except Exception as e:
        processing_status[job_id]["summary"] = f"Error: {str(e)}"
        processing_status[job_id]["summary_status"] = "error"
    finally:
        unload_llm()
        
def process_audio(file_path, job_id):
    processing_status[job_id] = {"progress": 0, "status": "Starting...", "result": None}
    start_time = time.time()
    try:
        processing_status[job_id] = {"progress": 5, "status": "Converting Audio..."}
        audio_path = file_path
        if not file_path.endswith(('.wav', '.mp3', '.flac', '.ogg')):
            temp_audio = os.path.join(tempfile.gettempdir(), f"{job_id}_converted.wav")
            if convert_to_audio(file_path, temp_audio):
                audio_path = temp_audio
            else:
                raise Exception("Conversion failed")
        data, sr = sf.read(audio_path, always_2d=True)
        waveform = torch.from_numpy(data.T).float()
        update_duration(job_id, int(len(data) / sr))
        processing_status[job_id].update({"progress": 20, "status": "Transcribing (Whisper)..."})
        asr = load_asr() 
        segments, _ = asr.transcribe(audio_path, beam_size=5, word_timestamps=True, vad_filter=True)
        whisper_segments = list(segments) 
        unload_asr() 
        processing_status[job_id].update({"progress": 60, "status": "Diarizing (Pyannote)..."})
        pipeline = load_pipeline() 
        annotation = perform_diarization(pipeline, waveform, sr)
        unload_pipeline() 
        if annotation is None:
            raise Exception("Diarization pipeline returned no data")
        processing_status[job_id].update({"progress": 80, "status": "Aligning Speakers..."})
        entries, speaker_map = merge_whisper_and_diarization(whisper_segments, annotation)
        merged_transcript, base_filename = save_outputs(job_id, os.path.basename(file_path), entries, speaker_map)
        time_taken = round(time.time() - start_time, 2)
        processing_status[job_id] = {
            "progress": 100,
            "status": "Completed!",
            "result": {
                "audio_file": os.path.basename(file_path),
                "transcript": merged_transcript,
                "speaker_map": speaker_map,
                "job_id": job_id,
                "base_filename": base_filename,
                "time_taken": time_taken
            },
            "summary_status": "pending"
        }
        summary_thread = threading.Thread(target=generate_summary_async, args=(job_id, merged_transcript, speaker_map, base_filename))
        summary_thread.start()
    except Exception as e:
        processing_status[job_id] = {"progress": 0, "status": f"Error: {str(e)}", "result": None}
        unload_asr()
        unload_pipeline()
        unload_llm()