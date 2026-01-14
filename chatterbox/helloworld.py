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

