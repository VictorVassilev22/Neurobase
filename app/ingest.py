import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain.text_splitter import Language

from app.config import EMBEDDING_MODEL, ALLOWED_EXTENSIONS, EXCLUDED_DIRS, get_vectorstore_dir


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTENSIONS

def collect_documents(root_path: str) -> List:
    root = Path(root_path)
    documents = []

    for file_path in root.rglob("*.*"):
        if not is_text_file(file_path):
            continue

        if any(part in EXCLUDED_DIRS for part in file_path.parts):
            print(f"⛔ Skipping {file_path} (excluded folder)")
            continue

        print(f"📄 Loading {file_path}...")

        try:
            loader = TextLoader(file_path, autodetect_encoding=True)
            for doc in loader.load():
                if not doc.page_content.strip():
                    print(f"⚠️ Skipping empty file: {file_path}")
                    continue
                doc.metadata["source"] = str(file_path)
                doc.page_content = f"# File name: {file_path.name}\n\n{doc.page_content}" # Add file name to content
                print(f"Doc metadata: {doc.metadata}")
                documents.append(doc)
        except Exception as e:
            print(f"⚠️ Skipping {file_path} — {type(e).__name__}: {e}")

    return documents


def run_ingest(code_path: str = "./"):
    print(f"📂 Ingesting from: {os.path.abspath(code_path)}")

    documents = collect_documents(code_path)
    # splitter = RecursiveCharacterTextSplitter.from_language(
    #     language=Language.PYTHON,
    #     chunk_size=256,
    #     chunk_overlap=64
    # )

    splitter = RecursiveCharacterTextSplitter(chunk_size=768, chunk_overlap=64)
    chunks = splitter.split_documents(documents)

    print(f"📄 {len(chunks)} chunks generated. Embedding...")

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})

    vectorstore_dir = get_vectorstore_dir(code_path)
    vectordb = Chroma.from_documents(chunks, embedding=embedder, persist_directory=vectorstore_dir)

    # vectordb.persist() // this line is redundant on ChromaDB >= 0.4.0
    print(f"✅ Embedded & stored to → {vectorstore_dir}/")

    # Write metadata
    with open("last_ingested.txt", "w", encoding="utf-8") as f:
        f.write(vectorstore_dir)
    print(f"📝 Ingestion path recorded to 'last_ingested.txt'")
