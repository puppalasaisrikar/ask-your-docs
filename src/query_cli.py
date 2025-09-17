# src/query_cli.py
import os
import pathlib
from dotenv import load_dotenv
from typing import List
from openai import OpenAI

from src.rag_utils import retrieve, format_context, build_index

APP_DIR = pathlib.Path(__file__).resolve().parents[1]

def ensure_env():
    load_dotenv(dotenv_path=APP_DIR / ".env")
    assert os.getenv("LLM_PROVIDER", "openai") in {"openai", "hf"}
    if os.getenv("LLM_PROVIDER", "openai") == "openai":
        assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY in .env"

def answer_with_openai(question: str, context: str) -> str:
    client = OpenAI()  # uses OPENAI_API_KEY from env
    prompt = f"""You are a helpful assistant for enterprise document Q&A.

Use ONLY the context to answer succinctly in 2–5 sentences and cite sources like [1], [2].
If the answer is unknown, say so.

Question:
{question}

Context:
{context}
"""
    # Using Responses API is fine too; keep it simple with chat.completions:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer with citations [1], [2] based on provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="(Re)build FAISS index from data/")
    parser.add_argument("-q", "--question", type=str, help="Question to ask")
    parser.add_argument("-k", "--topk", type=int, default=4, help="Top-k chunks to retrieve")
    args = parser.parse_args()

    ensure_env()

    if args.build:
        n, path = build_index()
        print(f"Indexed {n} chunks → {path}")

    if args.question:
        docs = retrieve(args.question, k=args.topk)
        ctx = format_context(docs)
        if os.getenv("LLM_PROVIDER", "openai") == "openai":
            ans = answer_with_openai(args.question, ctx)
        else:
            # Minimal fallback: print retrieved context only.
            ans = f"(LLM_PROVIDER != openai) Retrieved context:\n\n{ctx}"
        print("\n--- ANSWER ---\n")
        print(ans)
        print("\n--- SOURCES ---\n")
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", None)
            print(f"[{i}] {src}" + (f":p{page}" if page is not None else ""))

if __name__ == "__main__":
    main()
