import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import os, base64, time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from data_ingestion.embedder import embed_or_load
from retriever.rag_chain import build_rag_chain
from tts.tts_engine import generate_audio
from config import TTS_FILENAME, DATA_DIR

# Load .env variables (Groq API key)
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="🤖 Product Chatbot with RAG + Audio")
st.title("🤖 Product Chatbot with RAG + Audio")

session_id = "default"
if 'store' not in st.session_state:
    st.session_state.store = {}

# Load PDFs from data/
start_files = time.time()
pdf_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")])
print(f"⏱️ File loading took: {time.time() - start_files:.2f} sec")

if not pdf_files:
    st.warning("No PDF files found in the 'data/' folder.")
else:
    print(f"✅ Loaded {len(pdf_files)} PDF(s) from the data folder.")

    # Embedding stage
    start_embed = time.time()
    combined_bytes = b"".join([open(os.path.join(DATA_DIR, f), "rb").read() for f in pdf_files])
    vectorstore = embed_or_load(combined_bytes)
    retriever = vectorstore.as_retriever()
    print(f"⏱️ Embedding/vectorstore took: {time.time() - start_embed:.2f} sec")

    # LLM load
    start_llm = time.time()
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
    print(f"⏱️ LLM init took: {time.time() - start_llm:.2f} sec")

    # RAG chain creation
    start_rag = time.time()
    rag_chain = build_rag_chain(llm, retriever, st.session_state.store, session_id)
    print(f"⏱️ RAG chain setup took: {time.time() - start_rag:.2f} sec")

    query = st.text_input("Ask a question:")

    if query:
        # 🧹 Delete previous audio to avoid replaying old file
        if os.path.exists("outputs/response.mp3"):
            os.remove("outputs/response.mp3")

        # RAG response
        start_query = time.time()
        response = rag_chain.invoke({"input": query}, config={"configurable": {"session_id": session_id}})
        rag_time = time.time() - start_query
        st.markdown(f"**Answer:** {response['answer']}")
        st.info(f"🧠 RAG response time: {rag_time:.2f} sec")

        # TTS generation
        start_tts = time.time()
        audio_path = generate_audio(response['answer'])  # should return 'outputs/response.mp3'
        tts_time = time.time() - start_tts
        st.info(f"🔊 TTS generation time: {tts_time:.2f} sec")

        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as af:
                audio_bytes = af.read()

                if len(audio_bytes) > 1000:
                    b64 = base64.b64encode(audio_bytes).decode()
                    audio_html = f"""
                    <audio controls>
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    Your browser does not support the audio element.
                    </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Audio file is too small or invalid.")
        else:
            st.warning("❌ Audio generation failed.")
