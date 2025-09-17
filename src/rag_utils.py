# src/rag_utils.py
import os
import pathlib
from typing import List, Tuple

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings

# If you prefer OpenAI embeddings later, you can switch here.
# from langchain_openai import OpenAIEmbeddings  # requires `pip install langchain-openai`

# NEW: two embedding options
from langchain_openai import OpenAIEmbeddings  # OpenAI (stable)
# Optional HF path (later): pip install langchain-huggingface
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # modern import
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


# ---- Env & paths ----
APP_DIR = pathlib.Path(__file__).resolve().parents[1]  # project root
DATA_DIR = APP_DIR / "data"
INDEX_DIR = APP_DIR / "faiss_index"

def _load_env():
    load_dotenv(dotenv_path=APP_DIR / ".env")

def _get_embeddings():
    """Choose embeddings provider based on .env"""
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    if provider == "openai":
        # uses OPENAI_API_KEY from .env
        return OpenAIEmbeddings(model="text-embedding-3-small")  # cheap + good
    else:
        # fallback to HF if explicitly requested and available
        if not HF_AVAILABLE:
            raise RuntimeError(
                "HuggingFace embeddings requested but langchain-huggingface isn't installed. "
                "Run: pip install langchain-huggingface"
            )
        model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model)

def _load_documents() -> List:
    """Load PDFs and TXTs from data/."""
    docs = []
    for p in DATA_DIR.glob("**/*"):
        if p.is_file():
            try:
                if p.suffix.lower() == ".pdf":
                    docs.extend(PyPDFLoader(str(p)).load())
                elif p.suffix.lower() in {".txt", ".md"}:
                    docs.extend(TextLoader(str(p), encoding="utf-8").load())
            except Exception as e:
                print(f"[skip] {p.name}: {e}")
    return docs

def build_index(chunk_size: int = 800, chunk_overlap: int = 120) -> Tuple[int, str]:
    """Create FAISS index from documents under data/."""
    _load_env()
    docs = _load_documents()
    if not docs:
        raise RuntimeError("No documents found in data/. Add PDFs or TXT files first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    embeddings = _get_embeddings()

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(INDEX_DIR))

    return len(chunks), str(INDEX_DIR)

def load_index() -> FAISS:
    _load_env()
    embeddings = _get_embeddings()
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

def retrieve(question: str, k: int = 4):
    vs = load_index()
    return vs.similarity_search(question, k=k)

def format_context(docs: List) -> str:
    """Build a compact context block with source hints."""
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        tag = f"{src}" + (f":p{page}" if page is not None else "")
        parts.append(f"[{i}] ({tag})\n{d.page_content.strip()[:1000]}")  # trim per chunk
    return "\n\n".join(parts)

