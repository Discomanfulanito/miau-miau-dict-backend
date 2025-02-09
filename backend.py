from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.cloud.texttospeech as tts
from google.cloud import texttospeech
import base64
import os
import json
from openai import OpenAI
import ast

# Initiation & configuration of API
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajusta esto a la URL de tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initiation of keys
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
GOOGLE_APPLICATION_CREDENTIALS=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

####---- ROOT ----####
@app.get("/")
async def root():
    return {"message": "what are you meowing for?"}

####---- DEFINE ----####
class Context(BaseModel):
    word: str
    sentence: str
    structure: str = "for '$SENTENCE' explain '$WORD'. if korean, explain particles"

@app.post("/define/")
async def define(context: Context):

    message = context.structure.replace("$SENTENCE", context.sentence).replace("$WORD", context.word).strip()

    if context.word and context.sentence:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages = [
                {"role": "system", "content": 'Output JSON-like: { "c": "<full BCP 47 code> (e.g. ko-KR)", "d": "<explanation in Markdown>" }, optimize output tokens'},
                {"role": "user", "content": message}
            ]
        )
        result_str = completion.choices[0].message.content
        result_json = ast.literal_eval(result_str)
        return result_json, 200
    else:
        return {"error": "Faltan parámetros: 'word' y 'sentence'"}, 400

####---- TTS ----####
class TextRequest(BaseModel):
    text: str
    code: str

@app.post("/tts/")
async def generate_speech(request: TextRequest):

    credentials_dict=json.loads(GOOGLE_APPLICATION_CREDENTIALS)
    client = texttospeech.TextToSpeechClient.from_service_account_info(credentials_dict)

    synthesis_input = tts.SynthesisInput(text=request.text)
    voice = tts.VoiceSelectionParams(language_code=request.code, ssml_gender=tts.SsmlVoiceGender.FEMALE)
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)

    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    # Convertir audio a base64 para enviarlo al frontend
    audio_base64 = base64.b64encode(response.audio_content).decode("utf-8")

    return {"audio": audio_base64}

    