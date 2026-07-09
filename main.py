import os
import base64
import tempfile
import json
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq


app = FastAPI()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str



def safe_mode(arr):
    if len(arr) == 0:
        return None

    values, counts = np.unique(arr, return_counts=True)

    return float(values[np.argmax(counts)])



def calculate_stats(data):

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
        "correlation": []
    }


    if not data:
        return result


    columns = list(data[0].keys())

    numeric_columns = []


    for col in columns:

        values = []

        valid = True

        for row in data:
            try:
                values.append(float(row[col]))
            except:
                valid = False
                break


        if valid:
            numeric_columns.append(col)
            arr = np.array(values)

            result["columns"].append(col)

            result["mean"][col] = float(np.mean(arr))
            result["std"][col] = float(np.std(arr))
            result["variance"][col] = float(np.var(arr))
            result["min"][col] = float(np.min(arr))
            result["max"][col] = float(np.max(arr))
            result["median"][col] = float(np.median(arr))
            result["mode"][col] = safe_mode(arr)
            result["range"][col] = float(np.max(arr)-np.min(arr))

            result["value_range"][col] = {
                "min": float(np.min(arr)),
                "max": float(np.max(arr))
            }


        else:

            result["columns"].append(col)

            vals = [
                str(row[col])
                for row in data
            ]

            result["allowed_values"][col] = list(set(vals))



    # correlation matrix
    if len(numeric_columns) >= 2:

        matrix = []

        arrays = []

        for col in numeric_columns:
            arrays.append(
                [
                    float(row[col])
                    for row in data
                ]
            )


        corr = np.corrcoef(arrays)


        for row in corr:
            matrix.append(
                [
                    float(x)
                    for x in row
                ]
            )

        result["correlation"] = matrix


    return result



def parse_transcript(text):

    prompt = f"""
You are a data extraction engine.

Convert this speech transcript into a JSON dataset.

Transcript:
{text}

Return ONLY JSON.

Format:

{{
 "data":[
    {{"column1":value,"column2":value}}
 ]
}}

Rules:
- If there is no dataset, return:
{{"data":[]}}
- Preserve column names exactly.
- Extract every row.
"""


    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )


    content = response.choices[0].message.content


    try:
        return json.loads(content)["data"]

    except:

        return []



@app.post("/")
@app.post("/audio-analysis")
async def analyze_audio(req: AudioRequest):

    try:

        audio_bytes = base64.b64decode(
            req.audio_base64
        )


        with tempfile.NamedTemporaryFile(
            suffix=".wav"
        ) as audio_file:


            audio_file.write(audio_bytes)

            audio_file.flush()


            with open(
                audio_file.name,
                "rb"
            ) as f:

                transcription = client.audio.transcriptions.create(
                    file=f,
                    model="whisper-large-v3-turbo",
                    response_format="text"
                )


        dataset = parse_transcript(
            transcription
        )


        result = calculate_stats(
            dataset
        )


        return result



    except Exception as e:


        # always return valid schema

        return {
            "rows":0,
            "columns":[],
            "mean":{},
            "std":{},
            "variance":{},
            "min":{},
            "max":{},
            "median":{},
            "mode":{},
            "range":{},
            "allowed_values":{},
            "value_range":{},
            "correlation":[]
        }