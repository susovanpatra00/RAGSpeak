# import os, pickle
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from config import VECTOR_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME
# from utils.hashing import get_file_hash
# from data_ingestion.loader import load_pdf

# def embed_or_load(file_bytes: bytes):
#     file_hash = get_file_hash(file_bytes)
#     print(f"📂 Current file hash: {file_hash}\n")
#     vec_path = os.path.join(VECTOR_DIR, file_hash)  

#     if os.path.exists(os.path.join(vec_path, "index.faiss")) and os.path.exists(os.path.join(vec_path, "index.pkl")):
#         return FAISS.load_local(vec_path, HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME), allow_dangerous_deserialization=True)

#     docs = load_pdf(file_bytes)
#     splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
#     chunks = splitter.split_documents(docs)
#     embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
#     vectorstore = FAISS.from_documents(chunks, embedder)

#     vectorstore.save_local(vec_path)  # will save index.faiss + index.pkl
#     return vectorstore















import os, pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import VECTOR_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME
from utils.hashing import get_file_hash
from data_ingestion.loader import load_pdf
import streamlit as st

@st.cache_resource(show_spinner="🔄 Loading vectorstore from disk...")
def load_vectorstore(vec_path: str):
    print("📦 Loading existing vectorstore...")
    return FAISS.load_local(
        vec_path,
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME),
        allow_dangerous_deserialization=True
    )

def embed_or_load(file_bytes: bytes):
    file_hash = get_file_hash(file_bytes)
    vec_path = os.path.join(VECTOR_DIR, f"{file_hash}.faiss")

    print(f"📂 Current file hash: {file_hash}")
    print(f"📄 Vectorstore path: {vec_path}")

    if os.path.exists(vec_path):
        return load_vectorstore(vec_path)

    print("🧠 Vectorstore not found... generating embeddings.")
    docs = load_pdf(file_bytes)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"🔍 Split into {len(chunks)} chunks.")

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embedder)

    vectorstore.save_local(vec_path)
    print("✅ Embeddings saved.")

    return vectorstore
