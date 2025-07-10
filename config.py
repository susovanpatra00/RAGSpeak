import os

VECTOR_DIR = "vectorstore"
OUTPUT_DIR = "outputs"
DATA_DIR = os.path.join("data", "pdf")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TTS_FILENAME = os.path.join(OUTPUT_DIR, "response.wav")
