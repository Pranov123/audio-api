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

    return clean_float(
        values[np.argmax(counts)]
    )




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
        "correlation": []
    }

    if not data:
        return result

    columns = list(data[0].keys())

    numeric_columns = []

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

            numeric_columns.append(col)

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

    if len(numeric_columns) >= 2:

        arrays = []

        for c in numeric_columns:
            arrays.append([float(row[c]) for row in data])

        try:
            corr = np.corrcoef(arrays)

            matrix = []
            for r in corr:
                matrix.append([clean_float(v) for v in r])

            result["correlation"] = matrix

        except:
            result["correlation"] = []

    return result




def normalize_keys(data):
    """
    Normalizes column/key names so formatting differences in the transcript
    or LLM output (e.g. "점수 1" vs "점수1") don't produce mismatched keys.
    Currently collapses whitespace between '점수' and its trailing number.
    """

    normalized = []

    for row in data:

        new_row = {}

        for k, v in row.items():

            nk = k.strip()
            nk = re.sub(r"점수\s*(\d+)", r"점수\1", nk)
            nk = re.sub(r"\s+", "", nk) if re.fullmatch(r"점수\d+", nk) else nk

            new_row[nk] = v

        normalized.append(new_row)

    return normalized




def fallback_parser(text):

    print("TRANSCRIPT:", text)

    text = text.replace("\n", " ")

    label_numbers = sorted(set(int(n) for n in re.findall(r"점수\s*(\d+)", text)))

    print("DETECTED LABELS:", label_numbers)

    if label_numbers:

        columns = {}

        for i, label_num in enumerate(label_numbers):

            if i + 1 < len(label_numbers):
                next_num = label_numbers[i + 1]
                pattern = rf"점수\s*{label_num}[^0-9]*(.*?)(?=점수\s*{next_num})"
            else:
                pattern = rf"점수\s*{label_num}[^0-9]*(.*)"

            m = re.search(pattern, text, re.I)

            if m:
                nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", m.group(1))]
                if nums:
                    columns[f"점수{label_num}"] = nums

        lengths = [len(v) for v in columns.values()]

        if columns and all(l == lengths[0] and l > 0 for l in lengths):

            rows = []
            n = lengths[0]

            for i in range(n):
                rows.append({k: v[i] for k, v in columns.items()})

            print("PARSED ROWS:", rows)

            return rows

    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]

    if nums:
        rows = [{"값": x} for x in nums]
        print("PARSED ROWS (single column fallback):", rows)
        return rows

    return []



def parse_transcript(text):

    prompt = f"""
Extract the tabular dataset described in the transcript below.

Rules:
- Use the exact column/variable names as mentioned or implied in the transcript.
- Do NOT invent columns that aren't described.
- If the transcript only describes ONE variable/column, return data with only that one key per row.
- If it describes multiple variables, include all of them.
- Preserve the number of data points exactly as stated.
- Column names must not contain extra whitespace (e.g. use "점수1", not "점수 1").

Transcript:
{text}

Return ONLY JSON in this exact shape (keys/number of keys depend entirely on the transcript content):
{{"data": [ {{"<column_name>": <value>, "...": "..."}} ]}}

No explanation, no markdown, no code fences.
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content

        content = content.replace("```json", "").replace("```", "")

        start = content.find("{")
        end = content.rfind("}")

        if start != -1 and end != -1:

            obj = json.loads(content[start:end + 1])

            data = obj.get("data", [])

            if data and len(data) > 0:
                return normalize_keys(data)

            print("LLM FAILED, USING FALLBACK")
            return normalize_keys(fallback_parser(text))

    except Exception as e:
        print("LLM parse error:", e)

    return normalize_keys(fallback_parser(text))




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

                result = client.audio.transcriptions.create(
                    file=("audio.wav", audio.read()),
                    model="whisper-large-v3-turbo",
                    response_format="text"
                )

            if isinstance(result, str):
                transcript = result
            else:
                transcript = result.text

        finally:
            if os.path.exists(path):
                os.remove(path)

        data = parse_transcript(transcript)

        return calculate_statistics(data)

    except Exception as e:

        print("FINAL ERROR:", repr(e))

        return {
            "rows": 0,
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