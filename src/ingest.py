"""
Ingestion pipeline for the Medical Evidence RAG Agent.

Loads a JSON corpus of medical documents (PubMed abstracts or any
curated medical text), splits them into retrieval-friendly chunks,
embeds them with OpenAI embeddings, and persists a FAISS index to disk.

Expected input JSON schema (list of objects):
[
  {
    "id": "doc001",
    "pmid": "12345678",          # optional
    "title": "...",
    "source": "...",             # optional, e.g. journal name
    "url": "...",                # optional
    "text": "full abstract or document text"
  },
  ...
]

Usage:
    python src/ingest.py --input data/sample_medical_corpus.json --out index/faiss_medical
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# Local, free embedding model -- runs on-device via sentence-transformers,
# no API key or credits required. Must match the model used in rag_chain.py
# (a FAISS index is only valid for the embedding model that built it).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Corpus JSON must be a list of document objects.")
    required = {"id", "title", "text"}
    for i, doc in enumerate(data):
        missing = required - doc.keys()
        if missing:
            raise ValueError(f"Document at index {i} is missing fields: {missing}")
    return data


def build_documents(corpus: list[dict]) -> list[Document]:
    """Turn raw corpus entries into LangChain Documents with rich metadata.

    Metadata is what allows us to trace every retrieved chunk back to its
    original source (doc id, PMID, title, URL) for citation grounding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    for doc in corpus:
        chunks = splitter.split_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": doc["id"],
                        "pmid": doc.get("pmid", "N/A"),
                        "title": doc["title"],
                        "source": doc.get("source", "N/A"),
                        "url": doc.get("url", "N/A"),
                        "chunk_index": idx,
                    },
                )
            )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index from a medical corpus.")
    parser.add_argument(
        "--input",
        default="data/sample_medical_corpus.json",
        help="Path to corpus JSON file (default: bundled sample corpus).",
    )
    parser.add_argument(
        "--out",
        default="index/faiss_medical",
        help="Directory to save the FAISS index to.",
    )
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY is not set. Export it or add it to a .env file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading corpus from {args.input} ...")
    corpus = load_corpus(args.input)
    print(f"Loaded {len(corpus)} source documents.")

    documents = build_documents(corpus)
    print(f"Split into {len(documents)} retrieval chunks.")

    print(f"Embedding chunks with local model '{EMBEDDING_MODEL}' and building FAISS index ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(documents, embeddings)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(out_path))
    print(f"FAISS index saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
