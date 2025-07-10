# 🗣️ RAGSpeak — Terminal Chatbot with Audio + RAG

**RAGSpeak** is a fun personal project that combines Retrieval-Augmented Generation (RAG) with voice-based responses in the terminal. It loads PDF documents, answers user queries based on their content, and responds with both text and audio.

## 💡 What It Does

- Loads local PDFs from the `data/pdf/` directory
- Embeds them using HuggingFace sentence embeddings + FAISS
- Answers user questions using a Groq LLM (e.g., LLaMA-3)
- Uses conversational memory (chat history-aware responses)
- Speaks responses aloud using [Coqui TTS](https://github.com/coqui-ai/TTS)

---

## 📁 Project Structure & Key Files

| File/Folder                    | Purpose |
|-------------------------------|---------|
| `app.py`                      | Entry point. Runs the terminal chatbot. |
| `data_ingestion/embedder.py` | Embeds PDFs or loads from cached FAISS index. |
| `data_ingestion/loader.py`   | Loads PDF documents using LangChain loader. |
| `retriever/rag_chain.py`     | Builds a history-aware RAG chain using prompts. |
| `tts/tts_engine.py`          | Converts text answers to audio using Coqui TTS. |
| `config.py`                  | Stores paths and constants used across modules. |
| `data/pdf/`                  | Folder to place your PDFs. All are embedded and used. |
| `outputs/response.wav`       | Audio output is saved here and played on macOS. |
| `.env`                       | Stores your `GROQ_API_KEY` for the LLM. |

---

## 🔧 Requirements

- Python 3.11+
- macOS (uses `afplay` for audio playback)
- Groq API key (`.env` should contain `GROQ_API_KEY=your_key`)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
    ```

---

## ⚠️ Notes
* Virtual environment (`env311/`) is **excluded via `.gitignore`**.
* No frontend — just terminal I/O + audio output.

---

## 🚀 Run It

```bash
python app.py
```

---
