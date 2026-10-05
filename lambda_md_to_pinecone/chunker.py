
import logging

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


logger = logging.getLogger(__name__)


# =========================================================
# Text Splitter
# =========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)


# =========================================================
# Chunk Documents
# =========================================================

def chunk_documents(documents):

    chunks = splitter.split_documents(
        documents
    )

    logger.info(
        "Created %d chunks",
        len(chunks)
    )

    return chunks