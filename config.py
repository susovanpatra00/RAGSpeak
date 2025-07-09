import os

VECTOR_DIR = "vectorstore"
OUTPUT_DIR = "outputs"
DATA_DIR = "data"
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 500
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TTS_FILENAME = os.path.join(OUTPUT_DIR, "response.mp3")
