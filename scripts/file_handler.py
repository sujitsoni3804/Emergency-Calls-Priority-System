import os
import json
import subprocess
from scripts.config import OUTPUT_FOLDER

def convert_to_audio(input_path, output_path):
    try:
        subprocess.run(['ffmpeg', '-i', input_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_path],
                      check=True, capture_output=True)
        return True
    except:
        return False

def merge_speaker_segments(transcript):
    if not transcript:
        return []
    merged = []
    current_speaker = transcript[0]['speaker']
    current_text = transcript[0]['text']
    current_start = transcript[0]['start']
    current_end = transcript[0]['end']
    for i in range(1, len(transcript)):
        if transcript[i]['speaker'] == current_speaker:
            current_text += " " + transcript[i]['text']
            current_end = transcript[i]['end']
        else:
            merged.append({
                'speaker': current_speaker,
                'start': current_start,
                'end': current_end,
                'text': current_text.strip()
            })
            current_speaker = transcript[i]['speaker']
            current_text = transcript[i]['text']
            current_start = transcript[i]['start']
            current_end = transcript[i]['end']
    merged.append({
        'speaker': current_speaker,
        'start': current_start,
        'end': current_end,
        'text': current_text.strip()
    })
    return merged

def format_for_llm(transcript, speaker_map):
    formatted = []
    for item in transcript:
        speaker_label = speaker_map[item['speaker']]
        formatted.append({speaker_label: item['text']})
    return formatted

def save_outputs(job_id, audio_filename, transcript, speaker_map):
    merged_transcript = merge_speaker_segments(transcript)
    name_parts = os.path.splitext(audio_filename)
    base_name = name_parts[0]
    if '_' in base_name:
        parts = base_name.split('_', 1)
        if len(parts[0]) == 36 and parts[0].count('-') == 4:
            base_name = parts[1]
    json_data = format_for_llm(merged_transcript, speaker_map)
    json_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_{job_id}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    txt_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_{job_id}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for item in merged_transcript:
            speaker_label = speaker_map[item['speaker']]
            f.write(f"{speaker_label}: {item['text']}\n\n")
    return merged_transcript, base_name