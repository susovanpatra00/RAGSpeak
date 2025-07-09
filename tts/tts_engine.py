from gtts import gTTS
from config import TTS_FILENAME

def generate_audio(text: str) -> str:
    tts = gTTS(text)
    tts.save(TTS_FILENAME)
    return TTS_FILENAME
