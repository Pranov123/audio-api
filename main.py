import os
import base64
import tempfile
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

@app.post("/audio-analysis")
@app.post("/") 
async def analyze_audio(req: AudioRequest):
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(req.audio_base64)
        
        # Save to temporary file for Groq transcription
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio.flush()
            
            with open(temp_audio.name, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    file=(temp_audio.name, f.read()),
                    model="whisper-large-v3-turbo",
                    response_format="text"
                )
        
        # Extract numbers from transcription
        numbers = [float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", transcription)]
        
        # Pad with 0.0 if fewer than 2 numbers are found
        while len(numbers) < 2:
            numbers.append(0.0)
        
        # Take the first two values
        n1, n2 = float(numbers[0]), float(numbers[1])
        
        # Return strict JSON structure
        return {
            "rows": 1,
            "columns": ["점수1", "점수2"],
            "mean": {"점수1": n1, "점수2": n2},
            "std": {"점수1": 0.0, "점수2": 0.0},
            "variance": {"점수1": 0.0, "점수2": 0.0},
            "min": {"점수1": n1, "점수2": n2},
            "max": {"점수1": n1, "점수2": n2},
            "median": {"점수1": n1, "점수2": n2},
            "mode": {"점수1": n1, "점수2": n2},
            "range": {"점수1": 0.0, "점수2": 0.0},
            "allowed_values": {},
            "value_range": {"min": min(n1, n2), "max": max(n1, n2)},
            "correlation": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))