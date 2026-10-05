import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


load_dotenv()


# =========================================================
# Configuration
# =========================================================

PINECONE_API_KEY = os.environ[
    "PINECONE_API_KEY"
]

PINECONE_INDEX = os.environ.get(
    "PINECONE_INDEX",
    "rag-index"
)


# =========================================================
# Pinecone
# =========================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)


# =========================================================
# Check existing indexes
# =========================================================

indexes = pc.list_indexes()

existing_indexes = [
    index["name"]
    for index in indexes
]


print(
    "Existing indexes:",
    existing_indexes
)


# =========================================================
# Create index
# =========================================================

if PINECONE_INDEX in existing_indexes:

    print(
        f"Index already exists: "
        f"{PINECONE_INDEX}"
    )

else:

    print(
        f"Creating index: "
        f"{PINECONE_INDEX}"
    )

    pc.create_index(
        name=PINECONE_INDEX,

        # text-embedding-3-small
        dimension=1536,

        metric="cosine",

        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

    print(
        "Index creation requested."
    )
