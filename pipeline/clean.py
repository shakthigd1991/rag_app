# pipeline/clean.py
import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

def load_documents(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return loader.load()