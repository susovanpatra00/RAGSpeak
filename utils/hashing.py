# import hashlib

# def get_file_hash(file_bytes: bytes) -> str:
#     return hashlib.sha256(file_bytes).hexdigest()



from hashlib import sha256
from data_ingestion.loader import load_pdf

def get_file_hash(file_bytes: bytes) -> str:
    docs = load_pdf(file_bytes)
    content = "".join([d.page_content for d in docs])
    file_hash = sha256(content.encode("utf-8")).hexdigest()
    return file_hash
