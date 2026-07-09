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
        x=float(x)

        if math.isnan(x) or math.isinf(x):
            return 0.0

        return x

    except:
        return 0.0




def safe_mode(arr):

    if len(arr)==0:
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



    columns=list(data[0].keys())


    numeric_columns=[]



    for col in columns:


        values=[]

        ok=True


        for row in data:

            try:

                values.append(
                    float(row[col])
                )

            except:

                ok=False
                break



        if ok:


            numeric_columns.append(col)

            arr=np.array(
                values,
                dtype=float
            )


            result["columns"].append(col)


            result["mean"][col]=clean_float(
                np.mean(arr)
            )

            result["std"][col]=clean_float(
                np.std(arr)
            )

            result["variance"][col]=clean_float(
                np.var(arr)
            )

            result["min"][col]=clean_float(
                np.min(arr)
            )

            result["max"][col]=clean_float(
                np.max(arr)
            )

            result["median"][col]=clean_float(
                np.median(arr)
            )

            result["mode"][col]=safe_mode(arr)


            result["range"][col]=clean_float(
                np.max(arr)-np.min(arr)
            )


            result["value_range"][col]={

                "min":clean_float(np.min(arr)),

                "max":clean_float(np.max(arr))

            }



        else:


            result["columns"].append(col)


            vals=[
                str(row[col])
                for row in data
            ]


            result["allowed_values"][col]=list(
                set(vals)
            )



    # safe correlation

    if len(numeric_columns)>=2:


        matrix=[]


        arrays=[]


        for c in numeric_columns:

            arrays.append(

                [
                    float(row[c])
                    for row in data
                ]

            )



        try:

            corr=np.corrcoef(arrays)


            for r in corr:

                matrix.append(

                    [
                        clean_float(v)
                        for v in r
                    ]

                )


            result["correlation"]=matrix


        except:

            result["correlation"]=[]



    return result





def fallback_parser(text):

    print("TRANSCRIPT:", text)


    # normalize
    text = text.replace("\n", " ")


    # Find score1 numbers
    score1_patterns = [
        r"점수1[^0-9]*(.*?)(?=점수2)",
        r"점수\s*1[^0-9]*(.*?)(?=점수\s*2)",
        r"첫\s*번째\s*점수[^0-9]*(.*?)(?=두\s*번째)",
    ]


    score2_patterns = [
        r"점수2[^0-9]*(.*)",
        r"점수\s*2[^0-9]*(.*)",
        r"두\s*번째\s*점수[^0-9]*(.*)",
    ]


    a=[]
    b=[]


    for p in score1_patterns:

        m=re.search(
            p,
            text,
            re.I
        )

        if m:

            a=[
                float(x)
                for x in re.findall(
                    r"\d+(?:\.\d+)?",
                    m.group(1)
                )
            ]

            break



    for p in score2_patterns:

        m=re.search(
            p,
            text,
            re.I
        )

        if m:

            b=[
                float(x)
                for x in re.findall(
                    r"\d+(?:\.\d+)?",
                    m.group(1)
                )
            ]

            break



    print("PARSED SCORE1:",a)
    print("PARSED SCORE2:",b)



    if a and b:

        rows=[]

        for x,y in zip(a,b):

            rows.append(
                {
                    "점수1":x,
                    "점수2":y
                }
            )

        return rows



    # Last fallback:
    # Assume all numbers are two columns

    nums=[
        float(x)
        for x in re.findall(
            r"\d+(?:\.\d+)?",
            text
        )
    ]


    if len(nums)>=2:


        half=len(nums)//2


        first=nums[:half]
        second=nums[half:]


        rows=[]


        for x,y in zip(first,second):

            rows.append(
                {
                    "점수1":x,
                    "점수2":y
                }
            )


        if rows:
            return rows



    return []
def parse_transcript(text):


    prompt=f"""

Extract the dataset.

Transcript:

{text}


Return ONLY JSON:

{{
 "data":[
   {{
    "점수1":80,
    "점수2":90
   }}
 ]
}}

No explanation.
"""


    try:


        response=client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            temperature=0,

            messages=[

                {
                    "role":"user",
                    "content":prompt
                }

            ]

        )


        content=response.choices[0].message.content


        content=content.replace(
            "```json",
            ""
        ).replace(
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


            if data and len(data)>0:
                return data


            print("LLM FAILED, USING FALLBACK")
            return fallback_parser(text)


    except Exception as e:

        print("LLM parse error:",e)



    return fallback_parser(text)







@app.post("/")
@app.post("/audio-analysis")
async def analyze_audio(req:AudioRequest):


    try:


        audio_bytes=base64.b64decode(
            req.audio_base64
        )


        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as f:


            f.write(audio_bytes)

            path=f.name



        try:


            with open(path,"rb") as audio:


                result=client.audio.transcriptions.create(

                    file=(

                        "audio.wav",

                        audio.read()

                    ),

                    model="whisper-large-v3-turbo",

                    response_format="text"

                )



            if isinstance(result,str):

                transcript=result

            else:

                transcript=result.text



        finally:


            if os.path.exists(path):

                os.remove(path)



        data=parse_transcript(
            transcript
        )


        return calculate_statistics(
            data
        )



    except Exception as e:


        print("FINAL ERROR:",repr(e))


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