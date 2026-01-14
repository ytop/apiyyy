# apiyyy
It is an API service which can get the request of text and sample wav file(Optional) , then convert text to wav file.

The service sample code is like,

`
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Multilingual model (23 languages)
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

# English (use language_id="en")
text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text, language_id="en")
ta.save("test-english.wav", wav, model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)
`



# technical stack
fastapi, installed already

# init from main
> Created main.py with a /tts endpoint that:

- Accepts text (required) and language_id (default: "en") as form fields
- Accepts optional audio_prompt wav file for voice cloning
- Returns the generated wav file

Run the server with:
bash
uvicorn main:app --reload


Example usage:
bash
# Basic TTS
curl -X POST "http://localhost:8000/tts" -F "text=Hello world" -o output.wav

# With custom voice
curl -X POST "http://localhost:8000/tts" -F "text=Hello world" -F "audio_prompt=@sample.wav" -o output.wav
