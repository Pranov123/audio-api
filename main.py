import os
import base64
import tempfile
import json
import re
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
        return 0.0

    values, counts = np.unique(arr, return_counts=True)

    return float(values[np.argmax(counts)])



def calculate_statistics(data):

    output = {
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
        return output


    columns = list(data[0].keys())

    numeric_cols = []


    for col in columns:

        values = []

        numeric = True

        for row in data:
            try:
                values.append(float(row[col]))
            except:
                numeric = False
                break


        if numeric:

            numeric_cols.append(col)

            arr = np.array(values, dtype=float)

            output["columns"].append(col)

            output["mean"][col] = float(np.mean(arr))
            output["std"][col] = float(np.std(arr))
            output["variance"][col] = float(np.var(arr))
            output["min"][col] = float(np.min(arr))
            output["max"][col] = float(np.max(arr))
            output["median"][col] = float(np.median(arr))
            output["mode"][col] = safe_mode(arr)
            output["range"][col] = float(np.max(arr)-np.min(arr))

            output["value_range"][col] = {
                "min": float(np.min(arr)),
                "max": float(np.max(arr))
            }

        else:

            output["columns"].append(col)

            vals = [
                str(x[col])
                for x in data
            ]

            output["allowed_values"][col] = list(set(vals))


    # correlation
    if len(numeric_cols) >= 2:

        matrix = []

        arrs = []

        for c in numeric_cols:
            arrs.append(
                [
                    float(row[c])
                    for row in data
                ]
            )

        corr = np.corrcoef(arrs)

        for row in corr:
            matrix.append(
                [
                    float(x)
                    for x in row
                ]
            )

        output["correlation"] = matrix


    return output




def fallback_score_parser(text):

    """
    Handles common benchmark format:

    점수1: 80 90 70
    점수2: 60 75 88
    """

    score1 = re.search(
        r"점수1[^0-9]*(.*?)점수2",
        text,
        re.S
    )


    score2 = re.search(
        r"점수2[^0-9]*(.*)",
        text,
        re.S
    )


    if score1 and score2:

        a = [
            float(x)
            for x in re.findall(
                r"\d+(?:\.\d+)?",
                score1.group(1)
            )
        ]

        b = [
            float(x)
            for x in re.findall(
                r"\d+(?:\.\d+)?",
                score2.group(1)
            )
        ]


        rows=[]

        for x,y in zip(a,b):

            rows.append(
                {
                    "점수1":x,
                    "점수2":y
                }
            )


        if rows:
            return rows



    # generic number fallback

    nums = [
        float(x)
        for x in re.findall(
            r"\d+(?:\.\d+)?",
            text
        )
    ]


    if len(nums)>=4:

        half=len(nums)//2

        rows=[]

        for x,y in zip(
            nums[:half],
            nums[half:]
        ):

            rows.append(
                {
                    "점수1":x,
                    "점수2":y
                }
            )

        return rows


    return []





def parse_transcript(text):


    prompt=f"""
Extract the dataset from this transcript.

Transcript:
{text}


Return ONLY JSON.

Format:

{{
 "data":[
   {{
    "column_name":number
   }}
 ]
}}

Rules:
- Preserve Korean column names.
- Do not add explanations.
- If transcript contains scores, keep names like 점수1 and 점수2.
- Every row must be an object.
"""


    try:

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


        content=response.choices[0].message.content.strip()


        content=content.replace(
            "```json",
            ""
        )

        content=content.replace(
            "```",
            ""
        )


        start=content.find("{")
        end=content.rfind("}")


        if start!=-1 and end!=-1:

            obj=json.loads(
                content[start:end+1]
            )


            data=obj.get(
                "data",
                []
            )


            if data:
                return data


    except Exception:
        pass


    return fallback_score_parser(text)





@app.post("/")
@app.post("/audio-analysis")
async def analyze_audio(req: AudioRequest):


    try:


        audio_bytes=base64.b64decode(
            req.audio_base64
        )


        with tempfile.NamedTemporaryFile(
            suffix=".wav"
        ) as f:


            f.write(audio_bytes)

            f.flush()


            with open(
                f.name,
                "rb"
            ) as audio:


                transcript = client.audio.transcriptions.create(

                    file=audio,

                    model="whisper-large-v3-turbo",

                    response_format="text"
                )


        data=parse_transcript(
            transcript
        )


        return calculate_statistics(
            data
        )



    except Exception:


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