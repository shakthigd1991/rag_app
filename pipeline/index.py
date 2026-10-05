from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    EMBED_MODEL,
)


# =========================================================
# Embeddings
# =========================================================

embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    api_key=OPENAI_API_KEY,
)


# =========================================================
# Pinecone client
# =========================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)


# =========================================================
# Verify index
# =========================================================

existing_indexes = [
    index["name"]
    for index in pc.list_indexes()
]

if PINECONE_INDEX not in existing_indexes:

    raise RuntimeError(
        f"Pinecone index '{PINECONE_INDEX}' "
        f"does not exist. Available indexes: "
        f"{existing_indexes}"
    )


# =========================================================
# Vector store
# =========================================================

vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX,
    embedding=embeddings,
)
