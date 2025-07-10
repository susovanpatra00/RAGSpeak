from langchain_community.document_loaders import PyPDFLoader

def load_pdf_from_path(pdf_path: str) -> list:
    loader = PyPDFLoader(pdf_path)
    return loader.load()
