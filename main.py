import os
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
import io

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

@app.post("/audio-analysis")
async def analyze_audio(req: AudioRequest):
    try:
        # Base64をデコードして一時ファイルとして保存（Groq Whisper API用）
        audio_data = base64.b64decode(req.audio_base64)
        filename = f"temp_{req.audio_id}.wav"
        with open(filename, "wb") as f:
            f.write(audio_data)

        # Groq Whisper APIで文字起こし
        with open(filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3-turbo",
                response_format="text"
            )
        
        os.remove(filename)

        # ここで文字起こし結果から数値を抽出（サンプルロジック）
        # ※実際の仕様に合わせて抽出ロジックを実装してください
        numbers = [float(s) for s in transcription.split() if s.replace('.','',1).isdigit()]
        
        if not numbers:
            numbers = [0.0]

        # 統計データの計算
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