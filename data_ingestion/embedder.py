import os, pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import VECTOR_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME
from utils.hashing import get_file_hash
from data_ingestion.loader import load_pdf

def embed_or_load(file_bytes: bytes):
    file_hash = get_file_hash(file_bytes)
    vec_path = os.path.join(VECTOR_DIR, f"{file_hash}.faiss")

    if os.path.exists(vec_path):
        return FAISS.load_local(vec_path, HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME), allow_dangerous_deserialization=True)

    docs = load_pdf(file_bytes)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embedder)

    vectorstore.save_local(vec_path)
    return vectorstore
