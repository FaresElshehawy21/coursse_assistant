from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")
DB_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("Put your PDF files in the data/ folder first.")

    docs = []
    for pdf in pdfs:
        docs.extend(PyPDFLoader(str(pdf)).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        chunks,
        HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        persist_directory=DB_DIR,
    )
    print(f"Indexed {len(chunks)} chunks from {len(pdfs)} PDF(s).")


if __name__ == "__main__":
    main()
