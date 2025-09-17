import os
import pathlib
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI


from src.rag_utils import build_index, retrieve, format_context

APP_DIR = pathlib.Path(__file__).resolve().parent
load_dotenv(dotenv_path=APP_DIR / ".env")

st.set_page_config(page_title="Generative AI Knowledge Assistant", layout="wide")
st.title("Generative AI Knowledge Assistant (RAG)")

with st.expander("Environment Check", expanded=False):
    st.write("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
    st.write("HUGGINGFACEHUB_API_TOKEN set:", bool(os.getenv("HUGGINGFACEHUB_API_TOKEN")))
    st.write("EMBEDDING_MODEL:", os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    st.write("LLM_PROVIDER:", os.getenv("LLM_PROVIDER", "openai"))

st.markdown("### 1) Index your documents")
col1, col2 = st.columns([1,1])
with col1:
    uploaded = st.file_uploader("Upload PDFs or TXT files (optional, they will be saved to ./data)", type=["pdf","txt"], accept_multiple_files=True)
    if uploaded:
        data_dir = APP_DIR / "data"
        data_dir.mkdir(exist_ok=True)
        for f in uploaded:
            out = data_dir / f.name
            with open(out, "wb") as w:
                w.write(f.getbuffer())
        st.success(f"Saved {len(uploaded)} file(s) to data/.")

with col2:
    if st.button("Build/Refresh Index"):
        try:
            n_chunks, path = build_index()
            st.success(f"Indexed {n_chunks} chunks → {path}")
        except Exception as e:
            st.error(str(e))

st.markdown("### 2) Ask a question")
q = st.text_input("Your question", placeholder="e.g., What is our PTO policy? Provide citations.")
topk = st.slider("Top-k chunks", 2, 8, 4, 1)


def answer_with_openai(question: str, context: str) -> str:
    client = OpenAI()
    prompt = f"""You are a helpful assistant for enterprise document Q&A.

Use ONLY the context to answer succinctly in 2–5 sentences and cite sources like [1], [2].
If the answer is unknown, say so.

Question:
{question}

Context:
{context}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using only the provided context and include citations [1], [2]."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

if st.button("Search"):
    if not q.strip():
        st.warning("Please enter a question.")
    else:
        try:
            docs = retrieve(q, k=topk)
            ctx = format_context(docs)
            ans = answer_with_openai(q, ctx)
            st.markdown("#### Answer")
            st.write(ans)
            
            st.markdown("#### Sources")
            for i, d in enumerate(docs, 1):
                src = d.metadata.get("source", "unknown")
                page = d.metadata.get("page", None)
                st.write(f"[{i}] {src}" + (f":p{page}" if page is not None else ""))
                
            with st.expander("Context (debug)"):
                st.code(ctx)

        except Exception as e:
            st.error(str(e))



