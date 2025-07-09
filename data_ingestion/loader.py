from langchain_community.document_loaders import PyPDFLoader
from config import DATA_DIR

def load_pdf(file_bytes: bytes) -> list:
    file_path = f"{DATA_DIR}/temp.pdf"
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    loader = PyPDFLoader(file_path)
    return loader.load()
