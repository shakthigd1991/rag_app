import os
from .config import PINECONE_API_KEY, PINECONE_INDEX, EMBED_MODEL  # instead of hardcoding "text-embedding-3-small"
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX, dimension=1536, metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(PINECONE_INDEX)    
# Create vector store
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)  # Use the model from config.py
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


def upsert_chunks(chunks, document_id):
    # Add chunks WITHOUT wiping the index — accumulates a knowledge base.
    # For creating the metadata, we can use a unique document ID for each upload
    # Add chunks to vector store
    vector_store.add_texts(
        texts=[c.page_content for c in chunks],
        metadatas=[c.metadata for c in chunks],
        ids=[f"{document_id}_{i}" for i in range(len(chunks))]
    )
    return len(chunks)