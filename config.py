from dotenv import load_dotenv

load_dotenv()

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
# S3
# =========================================================

S3_BUCKET = os.environ["S3_BUCKET"]

S3_PREFIX = os.environ.get(
    "S3_PREFIX",
    "extracted-text/"
)


# =========================================================
# Embedding model
# =========================================================

EMBED_MODEL = os.environ.get(
    "EMBED_MODEL",
    "text-embedding-3-small"
)


# =========================================================
# Chat model
# =========================================================

CHAT_MODEL = os.environ.get(
    "CHAT_MODEL",
    "gpt-4o"
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
