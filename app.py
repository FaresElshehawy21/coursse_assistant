import base64
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

BG_PATH = Path(__file__).parent / "assets" / "bg.jpg"
if BG_PATH.exists():
    BG_B64 = base64.b64encode(BG_PATH.read_bytes()).decode()
else:
    BG_B64 = ""
    print("WARNING: assets/bg.jpg not found - running without background image")

DB_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "You are a friendly course assistant. Answer using ONLY the context below. "
    "Use short paragraphs and bullet points when helpful. "
    "If the answer is not in the context, say you could not find it in the course material.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

chain = prompt | llm | StrOutputParser()


def answer(message, history):
    docs = retriever.invoke(message)
    context = "\n\n".join(d.page_content for d in docs)

    partial = ""
    for token in chain.stream({"context": context, "question": message}):
        partial += token
        yield partial

    sources = sorted(
        {
            f"{Path(d.metadata.get('source', 'unknown')).name} (p.{d.metadata.get('page', 0) + 1})"
            for d in docs
        }
    )
    yield partial + "\n\n---\n📚 **Sources:** " + " · ".join(sources)


CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: auto;
    background: transparent !important;
}
body, gradio-app {
    background: linear-gradient(rgba(25, 14, 8, 0.55), rgba(25, 14, 8, 0.55)),
                url('data:image/jpeg;base64,BG_B64') center / cover fixed no-repeat !important;
}
h1, .prose h1 {color: #f5e6d3 !important; text-align: center;}
.prose p, .prose {color: #e8d3b9 !important; text-align: center;}
#chat {
    background: rgba(35, 20, 12, 0.72) !important;
    backdrop-filter: blur(6px);
    border: 1px solid rgba(214, 170, 120, 0.35) !important;
    border-radius: 14px !important;
}
#chat .prose, #chat .prose p {text-align: left;}
footer {visibility: hidden}
""".replace("BG_B64", BG_B64)

BROWN = gr.themes.Color(
    c50="#faf5f0", c100="#f1e4d6", c200="#e2c9ae", c300="#d1a97f", c400="#bd8656",
    c500="#a1673a", c600="#85512c", c700="#6b4024", c800="#55331f", c900="#44291c",
    c950="#2a1810", name="brown",
)

demo = gr.ChatInterface(
    answer,
    title="Courses Assistant",
    description="ANU Computer Science",
    chatbot=gr.Chatbot(
        elem_id="chat",
        height=520,
        placeholder="### 👋 Ask me anything about your lectures\nTry one of the examples below.",
    ),
    textbox=gr.Textbox(placeholder="Type your question here...", scale=7),
    examples=[
        "Summarize the main points of lecture 1",
        "Explain the key concepts in simple words",
        "Give me an example of the main topic in lecture 2",
        "What are the most important definitions?",
    ],
)

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue=BROWN, neutral_hue="stone"),
        css=CSS,
        js="() => { document.body.classList.add('dark'); }",
    )
