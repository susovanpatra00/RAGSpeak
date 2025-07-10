import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import DATA_DIR, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from data_ingestion.loader import load_pdf_from_path
from retriever.rag_chain import build_rag_chain
from tts.tts_engine import generate_audio

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# 🔊 macOS-specific audio player
def play_audio_mac(audio_path):
    if os.path.exists(audio_path):
        os.system(f"afplay '{audio_path}'")
    else:
        print("❌ Audio file not found.")

def main():
    print("🤖 Terminal Chatbot with RAG + Audio (macOS)")
    print(f"📁 Looking for PDFs in: {DATA_DIR}")

    # 🔍 Load all PDFs recursively
    pdf_files = sorted([str(p) for p in Path(DATA_DIR).rglob("*.pdf")])
    if not pdf_files:
        print("❌ No PDFs found.")
        return

    print(f"✅ Found {len(pdf_files)} PDF(s).")
    for f in pdf_files:
        print("  •", f)

    # 🧠 Load + combine documents
    all_docs = []
    for path in pdf_files:
        print(f"📄 Loading {path}")
        all_docs.extend(load_pdf_from_path(path))

    # 🔪 Chunk and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_docs)

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embedder)
    retriever = vectorstore.as_retriever()

    # 🔌 Load LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    # 🧠 Set up chat memory
    chat_history = ChatMessageHistory()
    rag_chain = build_rag_chain(llm, retriever, chat_history)

    # 🔁 Chat loop
    while True:
        query = input("\n🧠 Your question (type 'exit' to quit): ")
        if query.strip().lower() == "exit":
            break

        # response = rag_chain.invoke({"input": query})
        response = rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": "terminal"}}
        )

        answer = response['answer']
        print(f"\n📣 Answer: {answer}")

        print("🎙️ Generating voice...")
        audio_path = generate_audio(answer)
        play_audio_mac(audio_path)

if __name__ == "__main__":
    main()
