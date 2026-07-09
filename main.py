import os
import base64
import tempfile
import json
import re
import math
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY missing")

client = Groq(api_key=api_key)

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

def clean_float(x):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return 0.0
        return x
    except:
        return 0.0

def safe_mode(arr):
    if len(arr) == 0:
        return 0.0
    values, counts = np.unique(arr, return_counts=True)
    return clean_float(values[np.argmax(counts)])

def calculate_statistics(data):
    result = {
        "rows": len(data),
        "columns": [],
        "mean": {},
        "std": {},
        "variance": {},
        "min": {},
        "max": {},
        "median": {},
        "mode": {},
        "range": {},
        "allowed_values": {},
        "value_range": {},
        "correlation": [] # Kept strictly as an empty list
    }

    if not data:
        return result

    columns = list(data[0].keys())
    
    for col in columns:
        values = []
        ok = True
        for row in data:
            try:
                values.append(float(row[col]))
            except:
                ok = False
                break

        if ok:
            arr = np.array(values, dtype=float)
            result["columns"].append(col)
            result["mean"][col] = clean_float(np.mean(arr))
            result["std"][col] = clean_float(np.std(arr))
            result["variance"][col] = clean_float(np.var(arr))
            result["min"][col] = clean_float(np.min(arr))
            result["max"][col] = clean_float(np.max(arr))
            result["median"][col] = clean_float(np.median(arr))
            result["mode"][col] = safe_mode(arr)
            result["range"][col] = clean_float(np.max(arr) - np.min(arr))
            result["value_range"][col] = {
                "min": clean_float(np.min(arr)),
                "max": clean_float(np.max(arr))
            }
        else:
            result["columns"].append(col)
            vals = [str(row[col]) for row in data]
            result["allowed_values"][col] = list(set(vals))

    return result

def fallback_parser(text):
    text = text.replace("\n", " ")
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if len(nums) >= 2:
        half = len(nums) // 2
        rows = [{"점수1": nums[i], "점수2": nums[i + half]} for i in range(half)]
        return rows
    return []

def parse_transcript(text):
    prompt = f"Extract the dataset from this transcript. Return ONLY JSON: {{\"data\":[{{\"점수1\":80, \"점수2\":90}}]}}. Transcript: {text}"
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.replace("```json", "").replace("```", "")
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(content[start:end+1]).get("data", [])
            return data if data else fallback_parser(text)
    except:
        pass
    return fallback_parser(text)

@app.post("/")
@app.post("/audio-analysis")
async def analyze_audio(req: AudioRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            path = f.name
        try:
            with open(path, "rb") as audio:
                transcript = client.audio.transcriptions.create(
                    file=("audio.wav", audio.read()),
                    model="whisper-large-v3-turbo",
                    response_format="text"
                )
            transcript = transcript if isinstance(transcript, str) else transcript.text
        finally:
            if os.path.exists(path):
                os.remove(path)
        data = parse_transcript(transcript)
        return calculate_statistics(data)
    except Exception:
        return {
            "rows": 0, "columns": [], "mean": {}, "std": {}, "variance": {},
            "min": {}, "max": {}, "median": {}, "mode": {}, "range": {},
            "allowed_values": {}, "value_range": {}, "correlation": []
        }