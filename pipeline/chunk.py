# pipeline/chunk.py
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

def chunk_documents(documents, file_name):
    chunks = splitter.split_documents(documents)
    document_id = str(uuid.uuid4())
    for i, chunk in enumerate(chunks):
        chunk.metadata["document_id"] = document_id
        chunk.metadata["file_name"] = file_name
        chunk.metadata["chunk_number"] = i
    return chunks, document_id