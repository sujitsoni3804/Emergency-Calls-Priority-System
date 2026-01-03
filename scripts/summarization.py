import torch
import re
import os
from scripts.config import OUTPUT_FOLDER

def extract_urgency_rating(summary_text):
    match = re.search(r'Urgency Rating:\s*(\d)', summary_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

def clean_llm_response(full_response, prompt):
    response = full_response
    
    if "</think>" in response:
        response = response.split("</think>", 1)[1].strip()
    
    if prompt in response:
        response = response.replace(prompt, "").strip()
    
    lines = response.split('\n')
    summary_lines = []
    found_summary = False
    
    for line in lines:
        if line.strip().startswith('Summary:') or line.strip().startswith('Urgency Rating:'):
            if line.strip() not in summary_lines:
                summary_lines.append(line.strip())
                found_summary = True
    
    if found_summary:
        return '\n'.join(summary_lines)
    
    summary_parts = re.findall(r'(Summary:.*?(?=Summary:|Urgency Rating:|$))', response, re.DOTALL | re.IGNORECASE)
    urgency_parts = re.findall(r'(Urgency Rating:\s*\d)', response, re.IGNORECASE)
    
    if summary_parts and urgency_parts:
        clean_summary = summary_parts[0].strip()
        clean_urgency = urgency_parts[0].strip()
        return f"{clean_summary}\n{clean_urgency}"
    
    return response.strip()

def generate_summary(llm_tokenizer, llm_model, transcript, speaker_map):
    try:
        prompt_template = """You are an AI assistant analyzing a conversation transcript. Your task is to:
1. Provide a concise summary (under 100 words) describing the main topic and any emergency situation discussed.
2. Assess and rate the *urgency level* of the conversation on a scale from 0 to 5 based on the seriousness and immediacy of the emergency.

Use the following rating criteria:
- **0 No Emergency:** Routine or informational conversation. No distress, danger, or urgency. If there is no emergency condition found then strictly give 0 rating.
- **1 Minor Concern:** Mild issue or low-risk situation. No immediate action required.
- **2 Moderate Concern:** Some urgency or discomfort, but not life-threatening. Assistance may be needed soon.
- **3 High Concern:** Clearly urgent situation; requires prompt attention but not critical yet.
- **4 Critical Emergency:** Serious emergency where quick response is essential (e.g., severe injury, escalating danger).
- **5 Life-Threatening Emergency:** Extreme urgency involving imminent danger, life and death, or immediate medical or rescue action required.

Your response format should be strictly as follows:
Summary: <brief summary of conversation>
Urgency Rating: <0-5>

CONVERSATION TRANSCRIPT:
"""
        formatted_text = ""
        for item in transcript:
            speaker_label = speaker_map[item['speaker']]
            formatted_text += f"{speaker_label}: {item['text']}\n\n"
        
        full_prompt = f"{prompt_template}\n{formatted_text}"
        messages = [{"role": "user", "content": full_prompt}]
        prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=204800)
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=llm_tokenizer.eos_token_id,
                eos_token_id=llm_tokenizer.eos_token_id
            )
        
        full_response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_response = clean_llm_response(full_response, full_prompt)
        
        return final_response
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def save_summary(job_id, base_filename, summary):
    summary_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_{job_id}_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)