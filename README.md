# Course Material Assistant (RAG)

Ask questions about your course PDFs and get answers with cited sources.

**Stack:** LangChain (LCEL) · Chroma · HuggingFace embeddings · Groq (Llama 3.3) · Gradio

## Run
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your free Groq API key
3. Put PDFs in `data/`
4. `python ingest.py`
5. `python app.py`

## How it works
PDFs → chunks (800 chars, 120 overlap) → embeddings → Chroma → top-4 retrieval → LLM answers only from context → sources shown.
