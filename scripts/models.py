import os
import gc
import threading
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from scripts.config import HF_TOKEN, WHISPER_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, MODELS_FOLDER

pipeline = None
asr = None
llm_tokenizer = None
llm_model = None

_pipeline_refs = 0
_asr_refs = 0
_llm_refs = 0
_pipeline_pending_unload = False
_asr_pending_unload = False
_llm_pending_unload = False
_refs_lock = threading.Lock()

def clear_gpu():
    global pipeline, asr, llm_model, llm_tokenizer
    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
    finally:
        gc.collect()

def load_pipeline():
    global pipeline
    if pipeline is not None:
        return pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    if torch.cuda.is_available():
        try:
            pipeline.to(torch.device("cuda"))
        except Exception:
            pass
    return pipeline

def _force_unload_pipeline():
    global pipeline
    if pipeline is not None:
        try:
            del pipeline
        finally:
            pipeline = None
            clear_gpu()

def pipeline_acquire():
    global _pipeline_refs, _pipeline_pending_unload
    with _refs_lock:
        _pipeline_refs += 1
        if pipeline is None:
            load_pipeline()
        return pipeline

def pipeline_release():
    global _pipeline_refs, _pipeline_pending_unload
    with _refs_lock:
        if _pipeline_refs > 0:
            _pipeline_refs -= 1
        if _pipeline_refs == 0 and _pipeline_pending_unload:
            _pipeline_pending_unload = False
            _force_unload_pipeline()

def unload_pipeline():
    global _pipeline_refs, _pipeline_pending_unload
    with _refs_lock:
        if _pipeline_refs > 0:
            _pipeline_pending_unload = True
            return
    _force_unload_pipeline()

def load_asr():
    global asr
    if asr is not None:
        return asr
    asr = WhisperModel(WHISPER_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    return asr

def _force_unload_asr():
    global asr
    if asr is not None:
        try:
            del asr
        finally:
            asr = None
            clear_gpu()

def asr_acquire():
    global _asr_refs, _asr_pending_unload
    with _refs_lock:
        _asr_refs += 1
        if asr is None:
            load_asr()
        return asr

def asr_release():
    global _asr_refs, _asr_pending_unload
    with _refs_lock:
        if _asr_refs > 0:
            _asr_refs -= 1
        if _asr_refs == 0 and _asr_pending_unload:
            _asr_pending_unload = False
            _force_unload_asr()

def unload_asr():
    global _asr_refs, _asr_pending_unload
    with _refs_lock:
        if _asr_refs > 0:
            _asr_pending_unload = True
            return
    _force_unload_asr()

def load_llm():
    global llm_tokenizer, llm_model
    if llm_model is not None and llm_tokenizer is not None:
        return llm_tokenizer, llm_model
    model_path = os.path.join(MODELS_FOLDER, "gemma-3-4b-it")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    llm_model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, device_map="auto")
    return llm_tokenizer, llm_model

def _force_unload_llm():
    global llm_model, llm_tokenizer
    if llm_model is not None or llm_tokenizer is not None:
        try:
            if llm_model is not None:
                del llm_model
            if llm_tokenizer is not None:
                del llm_tokenizer
        finally:
            llm_model = None
            llm_tokenizer = None
            clear_gpu()

def llm_acquire():
    global _llm_refs, _llm_pending_unload
    with _refs_lock:
        _llm_refs += 1
        if llm_model is None or llm_tokenizer is None:
            load_llm()
        return llm_tokenizer, llm_model

def llm_release():
    global _llm_refs, _llm_pending_unload
    with _refs_lock:
        if _llm_refs > 0:
            _llm_refs -= 1
        if _llm_refs == 0 and _llm_pending_unload:
            _llm_pending_unload = False
            _force_unload_llm()

def unload_llm():
    global _llm_refs, _llm_pending_unload
    with _refs_lock:
        if _llm_refs > 0:
            _llm_pending_unload = True
            return
    _force_unload_llm()

def transcribe_audio(audio_path, language=None, **whisper_kwargs):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)
    asr_local = asr_acquire()
    try:
        result = asr_local.transcribe(audio_path, **whisper_kwargs)
        if isinstance(result, tuple) and len(result) >= 1:
            segments_raw = result[0]
        elif isinstance(result, dict) and "segments" in result:
            segments_raw = result["segments"]
        else:
            segments_raw = getattr(result, "segments", result)
        text_parts = []
        segments = []
        for seg in segments_raw:
            if isinstance(seg, dict):
                start_t = seg.get("start", None)
                end_t = seg.get("end", None)
                seg_text = seg.get("text", "")
            else:
                start_t = getattr(seg, "start", None)
                end_t = getattr(seg, "end", None)
                seg_text = getattr(seg, "text", str(seg))
            segments.append({"start": start_t, "end": end_t, "text": seg_text})
            text_parts.append(seg_text)
        return {"text": " ".join(text_parts).strip(), "segments": segments}
    finally:
        asr_release()

def diarize_audio(audio_path):
    p = pipeline_acquire()
    try:
        diarization = p({"uri": os.path.basename(audio_path), "audio": audio_path})
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({"speaker": speaker, "start": float(turn.start), "end": float(turn.end)})
        return turns
    finally:
        pipeline_release()

def generate_summary(transcript_text, max_new_tokens=256, prompt_prefix=None):
    if not transcript_text or not transcript_text.strip():
        return ""
    tokenizer_local, model_local = llm_acquire()
    prompt = (prompt_prefix or "Summarize the following transcript in a concise, actionable paragraph:\n\n") + transcript_text
    try:
        inputs = tokenizer_local(prompt, return_tensors="pt", truncation=True)
        try:
            device = next(model_local.parameters()).device
        except Exception:
            device = torch.device("cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if getattr(tokenizer_local, "pad_token_id", None) is None:
            pad_id = getattr(tokenizer_local, "eos_token_id", None)
            if pad_id is not None:
                tokenizer_local.pad_token_id = pad_id
        with torch.no_grad():
            outputs = model_local.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if isinstance(outputs, (list, tuple)):
            outputs_tensor = outputs[0]
        else:
            outputs_tensor = outputs
        try:
            if outputs_tensor.dim() == 2:
                total_len = outputs_tensor.size(1)
                input_len = inputs["input_ids"].size(1)
                if total_len > input_len:
                    gen_tensor = outputs_tensor[:, input_len:]
                else:
                    gen_tensor = outputs_tensor[0]
            elif outputs_tensor.dim() == 1:
                gen_tensor = outputs_tensor
            else:
                gen_tensor = outputs_tensor.view(-1)
        except Exception:
            gen_tensor = outputs_tensor[0] if outputs_tensor.ndim > 0 else outputs_tensor
        try:
            ids = gen_tensor.cpu().numpy().tolist() if hasattr(gen_tensor, "cpu") else list(gen_tensor)
        except Exception:
            try:
                ids = gen_tensor.tolist()
            except Exception:
                ids = []
        decoded = tokenizer_local.decode(ids, skip_special_tokens=True).strip()
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):].strip()
        return decoded
    finally:
        llm_release()

def summarize_audio(audio_path, do_diarize=True, whisper_kwargs=None, summary_prompt=None):
    whisper_kwargs = whisper_kwargs or {}
    result = {"transcript": None, "diarization": None, "summary": None}
    trans = transcribe_audio(audio_path, **whisper_kwargs)
    result["transcript"] = trans
    if do_diarize:
        diar = diarize_audio(audio_path)
        result["diarization"] = diar
    text = trans.get("text", "") if isinstance(trans, dict) else str(trans)
    if not text.strip():
        return result
    summary = generate_summary(text, prompt_prefix=summary_prompt)
    result["summary"] = summary
    return result

def init_models():
    pass