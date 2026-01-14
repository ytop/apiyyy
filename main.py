import torch
import torchaudio as ta
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import tempfile
import os

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_pretrained(device=device)


@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    language_id: str = Form("en"),
    audio_prompt: UploadFile | None = File(None)
):
    prompt_path = None
    if audio_prompt:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(await audio_prompt.read())
            prompt_path = f.name

    wav = model.generate(text, language_id=language_id, audio_prompt_path=prompt_path)

    output_path = tempfile.mktemp(suffix=".wav")
    ta.save(output_path, wav, model.sr)

    if prompt_path:
        os.unlink(prompt_path)

    return FileResponse(output_path, media_type="audio/wav", filename="output.wav")
