from hashlib import sha256
from data_ingestion.loader import load_pdf_from_path

def get_file_hash_from_path(pdf_path: str) -> str:
    docs = load_pdf_from_path(pdf_path)
    content = "".join([d.page_content for d in docs])
    return sha256(content.encode("utf-8")).hexdigest()
