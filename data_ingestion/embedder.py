import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import VECTOR_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME
from utils.hashing import get_file_hash_from_path
from data_ingestion.loader import load_pdf_from_path

def embed_or_load(pdf_path: str):
    file_hash = get_file_hash_from_path(pdf_path)
    vec_path = os.path.join(VECTOR_DIR, f"{file_hash}.faiss")

    print(f"📂 File hash: {file_hash}")
    print(f"📄 Vectorstore path: {vec_path}")

    if os.path.exists(vec_path):
        print("✅ Vectorstore found. Loading...")
        return FAISS.load_local(vec_path, HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME), allow_dangerous_deserialization=True)

    print("🧠 Vectorstore not found... generating embeddings.")
    docs = load_pdf_from_path(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    print(f"📄 Split into {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  ➤ Chunk {i+1}: {chunk.page_content[:80]}...")

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embedder)
    vectorstore.save_local(vec_path)
    print("✅ Vectorstore saved.")
    return vectorstore
