import os
from TTS.api import TTS  # assuming Coqui TTS
from config import TTS_FILENAME  # Should be set to "outputs/response.mp3"

def generate_audio(text: str) -> str:
    # Initialize TTS model (if needed, cache globally later for speed)
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(TTS_FILENAME), exist_ok=True)

    # Always write to the same file
    tts.tts_to_file(text=text, file_path=TTS_FILENAME)
    print(f"✅ Audio saved to {TTS_FILENAME}")
    return TTS_FILENAME
