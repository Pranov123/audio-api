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




def is_strict_number(v):
    """
    True only if v is unambiguously a bare number (int/float, or a string
    that IS a number with nothing else attached - no units, no words,
    no trailing/leading characters). This is what prevents things like
    "3점", "우수", "약 3" from being silently treated as numeric.
    """

    if isinstance(v, bool):
        return False

    if isinstance(v, (int, float)):
        return True

    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return False
        # must match a plain int/float and NOTHING else (no suffix/prefix)
        return bool(re.fullmatch(r"-?\d+(\.\d+)?", s))

    return False




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

    # Use the UNION of keys across ALL rows, not just row 0, so a column
    # missing from the first row (but present elsewhere) isn't dropped.
    columns = []
    seen = set()

    for row in data:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                columns.append(k)

    numeric_columns = []

    for col in columns:

        values = []
        ok = True

        for row in data:
            v = row.get(col, None)

            # A column only counts as numeric if EVERY row's value is
            # strictly a bare number. Any non-numeric value (word, label,
            # a number with attached units/qualifiers, missing value, etc.)
            # disqualifies the whole column from being numeric.
            if not is_strict_number(v):
                ok = False
                break

            values.append(float(v))

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

            # tolerate rows that are missing this key entirely
            vals = [str(row.get(col, "")) for row in data]

            result["allowed_values"][col] = list(set(vals))

    # correlation only makes sense if:
    #  - there are 2+ numeric columns, AND
    #  - every numeric column has non-zero variance
    if len(numeric_columns) >= 2:

        arrays = []

        for c in numeric_columns:
            arrays.append([float(row[c]) for row in data])

        stds = [np.std(a) for a in arrays]

        if all(s > 0 for s in stds):

            try:
                corr = np.corrcoef(arrays)

                matrix = []
                for r in corr:
                    matrix.append([clean_float(v) for v in r])

                result["correlation"] = matrix

            except:
                result["correlation"] = []

        else:
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
    """
    Regex-based fallback used only when the LLM call fails or returns
    nothing. IMPORTANT: this must NOT force values into numbers when the
    original text segment contains anything other than a bare number
    (units, words, qualifiers, etc.) - otherwise categorical data gets
    mis-classified as numeric downstream.
    """

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
                # Extract candidate tokens (comma/space separated chunks),
                # not just raw digit substrings. Keep each token AS-IS so
                # that things like "3점" or "우수" survive intact instead
                # of being reduced to a bare digit.
                raw_segment = m.group(1).strip()

                tokens = [
                    t.strip() for t in re.split(r"[,\uFF0C/]|(?:\s{2,})", raw_segment)
                    if t.strip()
                ]

                if not tokens:
                    tokens = [t for t in raw_segment.split() if t]

                if tokens:
                    columns[f"점수{label_num}"] = tokens

        lengths = [len(v) for v in columns.values()]

        if columns and all(l == lengths[0] and l > 0 for l in lengths):

            rows = []
            n = lengths[0]

            for i in range(n):
                row = {}
                for k, v in columns.items():
                    token = v[i]
                    # Only coerce to a number if the token is a STRICT bare
                    # number. Otherwise keep the original string so
                    # calculate_statistics correctly routes it to
                    # allowed_values instead of forcing it numeric.
                    if is_strict_number(token):
                        row[k] = float(token)
                    else:
                        row[k] = token
                rows.append(row)

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
- EVERY row in "data" must include ALL the same keys, even if a value is 0. Never omit a key from a row just because its value is 0 or empty.
- Preserve the number of data points exactly as stated.
- Column names must not contain extra whitespace (e.g. use "점수1", not "점수 1").
- CRITICAL - preserve value type exactly as spoken:
  - If a value is stated as a plain number (digits), return it as a JSON number with nothing else attached.
  - If a value is a word, label, category, grade, rating name, or anything that is not literally a bare number (e.g. "우수", "보통", "합격", "3등급", "없음", "3점" with a unit word attached), return it as a JSON string EXACTLY as heard. Do NOT translate it into a number, do NOT infer a numeric equivalent, do NOT strip qualifier words attached to a number.
  - A column should only contain JSON numbers for ALL rows if every single row's value for that variable is unambiguously a bare number. If even one row's value is non-numeric text, return EVERY value in that column as a JSON string (do not silently convert the rest to numbers).
  - Never "clean up," round, or normalize a value's type based on what you think it should logically be - only reflect exactly what was said.

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