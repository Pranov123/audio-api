import os
import base64
import numpy as np
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

# Define both /audio-analysis AND / to catch the 404
@app.post("/audio-analysis")
@app.post("/") 
async def analyze_audio(req: AudioRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio.flush()
            
            with open(temp_audio.name, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    file=(temp_audio.name, f.read()),
                    model="whisper-large-v3-turbo",
                    response_format="text"
                )
        
        import re
        numbers = [float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", transcription)]
        if not numbers:
            numbers = [0.0]
        
        arr = np.array(numbers)
        
        return {
            "rows": len(arr),
            "columns": ["value"],
            "mean": {"value": float(np.mean(arr))},
            "std": {"value": float(np.std(arr))},
            "variance": {"value": float(np.var(arr))},
            "min": {"value": float(np.min(arr))},
            "max": {"value": float(np.max(arr))},
            "median": {"value": float(np.median(arr))},
            "mode": {"value": float(np.mean(arr))},
            "range": {"value": float(np.ptp(arr))},
            "allowed_values": {},
            "value_range": {"min": float(np.min(arr)), "max": float(np.max(arr))},
            "correlation": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))