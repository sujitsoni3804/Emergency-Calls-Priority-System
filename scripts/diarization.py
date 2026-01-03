import torch
import numpy as np
from pyannote.core import Segment

def perform_diarization(pipeline, waveform, sample_rate):
    try:
        if torch.is_tensor(waveform):
            waveform = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
        out = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        return out
    except Exception as e:
        print(f"Diarization error: {e}")
        return None
    
def merge_whisper_and_diarization(whisper_segments, annotation):
    final_segments = []
    for segment in whisper_segments:
        start = segment.start
        end = segment.end
        text = segment.text
        whisper_interval = Segment(start, end)
        best_speaker = "Unknown"
        max_overlap = 0
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            intersection = whisper_interval & turn
            overlap_duration = intersection.duration
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = speaker
        if best_speaker == "Unknown" and final_segments:
            best_speaker = final_segments[-1]['speaker']
        final_segments.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "speaker": best_speaker,
            "text": text.strip()
        })
    unique_speakers = sorted(list(set(s['speaker'] for s in final_segments)))
    speaker_map = {spk: f"Speaker {i+1}" for i, spk in enumerate(unique_speakers)}
    return final_segments, speaker_map