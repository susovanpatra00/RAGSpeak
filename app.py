import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from data_ingestion.embedder import embed_or_load
from retriever.rag_chain import build_rag_chain
from tts.tts_engine import generate_audio
from config import TTS_FILENAME, DATA_DIR
import base64

# Load .env variables (Groq API key)
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="🤖 Product Chatbot with RAG + Audio")
st.title("🤖 Product Chatbot with RAG + Audio")

# Default session ID and chat history store
session_id = "default"
if 'store' not in st.session_state:
    st.session_state.store = {}

# Loading all PDFs from the data folder
pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
if not pdf_files:
    st.warning("No PDF files found in the 'data/' folder.")
else:
    st.success(f"Loaded {len(pdf_files)} PDF(s) from the data folder.")

    # Combining all PDFs into one embedding
    combined_bytes = b"".join([open(os.path.join(DATA_DIR, f), "rb").read() for f in pdf_files])
    vectorstore = embed_or_load(combined_bytes)
    retriever = vectorstore.as_retriever()

    # Loading LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Creating conversational RAG chain
    rag_chain = build_rag_chain(llm, retriever, st.session_state.store, session_id)

    # Input box for user query
    query = st.text_input("Ask a question about the product:")

    if query:
        response = rag_chain.invoke({"input": query}, config={"configurable": {"session_id": session_id}})
        st.markdown(f"**Answer:** {response['answer']}")

        # audio_path = generate_audio(response['answer'])
        # with open(audio_path, "rb") as af:
        #     st.audio(af.read(), format="audio/mp3")



        # audio_path = generate_audio(response['answer'])
        #
        # if audio_path and os.path.exists(audio_path):
        #     with open(audio_path, "rb") as af:
        #         audio_bytes = af.read()
        #         st.audio(audio_bytes, format="audio/mp3")
        # else:
        #     st.warning("Audio generation failed or file not found.")


        audio_path = generate_audio(response['answer'])

        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as af:
                audio_bytes = af.read()

                if len(audio_bytes) > 1000:
                    # mac problem most probably
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


