import logging

from pinecone import Pinecone

from langchain_pinecone import (
    PineconeVectorStore
)

from embeddings import embeddings

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX
)


logger = logging.getLogger(__name__)


# =========================================================
# Pinecone Client
# =========================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)


# =========================================================
# Pinecone Index
# =========================================================

index = pc.Index(
    PINECONE_INDEX
)


# =========================================================
# LangChain Vector Store
# =========================================================

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)


# =========================================================
# Store Documents
# =========================================================

def store_documents(
    documents,
    ids
):

    vector_store.add_documents(
        documents=documents,
        ids=ids
    )

    logger.info(
        "Stored %d documents in Pinecone",
        len(documents)
    )

    return len(documents)