import os


# =========================================================
# OpenAI
# =========================================================

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


# =========================================================
# Pinecone
# =========================================================

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

PINECONE_INDEX = os.environ.get(
    "PINECONE_INDEX",
    "rag-index"
)


# =========================================================
# Embedding model
# =========================================================

EMBED_MODEL = os.environ.get(
    "EMBED_MODEL",
    "text-embedding-3-small"
)


# =========================================================
# Chunking
# =========================================================

CHUNK_SIZE = int(
    os.environ.get(
        "CHUNK_SIZE",
        "1500"
    )
)

CHUNK_OVERLAP = int(
    os.environ.get(
        "CHUNK_OVERLAP",
        "200"
    )
)