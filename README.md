# 📄 Ask-Your-Docs (GenAI Document Assistant)

**Ask-Your-Docs** is a GenAI-powered assistant that lets you upload any document (PDF, DOCX, TXT) and ask natural-language questions about it.  
It parses the text, generates embeddings, performs similarity search, and then uses a Large Language Model (LLM) to generate context-aware answers with supporting evidence.

---

## 🚀 Features
- 🔍 **Multi-format ingestion**: Supports PDF, DOCX, and TXT files  
- 🧩 **Chunked embeddings**: Splits documents intelligently and stores them in a vector index (FAISS/Chroma)  
- 🤖 **GenAI-powered Q&A**: Answers questions with direct references to source snippets  
- 🖥️ **Simple interface**: Lightweight web app (Streamlit/FastAPI/Flask)  
- 📑 **Evidence-based answers**: Returns both concise responses and retrieved text chunks for transparency  

---

## ⚡ Quickstart

1. **Clone the repo**
   ```bash
   git clone https://github.com/puppalasaisrikar/ask-your-docs.git
   cd ask-your-docs

2. Install dependencies:
pip install -r requirements.txt
(or use conda env create -f environment.yml if you prefer conda)

3. Run the app:
python src/app.py

4. Open your browser → http://localhost:8501 (if using Streamlit)
-Upload a document, then ask questions in plain English.

---

## 📊 Example
Input Document: example_contract.pdf

Question: "What is the penalty for late payment?"

Answer:
The contract specifies a 2% monthly penalty on late payments (see Section 5.2, Page 3).

---

## 🛠️ Tech Stack
Python 3.9+

FAISS / ChromaDB for vector search

OpenAI / Ollama / HuggingFace LLMs

Streamlit / FastAPI for interface

---

## 📜 License
This project is released under the MIT License.
Feel free to use, adapt, and improve!

---

## 🙌 Acknowledgements
Built as part of a GenAI hackathon project - demonstrating how LLMs + embeddings can make document understanding interactive and transparent.
