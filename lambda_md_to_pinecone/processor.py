import hashlib
import logging

from s3_loader import (
    load_markdown_from_s3
)

from chunker import (
    chunk_documents
)

from vector_store import (
    store_documents
)


logger = logging.getLogger(__name__)


# =========================================================
# Create deterministic document ID
# =========================================================

def create_document_id(
    bucket_name,
    key
):

    value = (
        f"{bucket_name}:{key}"
    )

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


# =========================================================
# Process Markdown
# =========================================================

def process_markdown(
    bucket_name,
    key
):

    logger.info(
        "Starting processing: s3://%s/%s",
        bucket_name,
        key
    )

    # -----------------------------------------------------
    # 1. Load
    # -----------------------------------------------------

    document = load_markdown_from_s3(
        bucket_name,
        key
    )

    # -----------------------------------------------------
    # 2. Chunk
    # -----------------------------------------------------

    chunks = chunk_documents(
        [document]
    )

    if not chunks:

        logger.warning(
            "No chunks created for %s",
            key
        )

        return 0

    # -----------------------------------------------------
    # 3. Create document ID
    # -----------------------------------------------------

    document_id = create_document_id(
        bucket_name,
        key
    )

    # -----------------------------------------------------
    # 4. Add metadata
    # -----------------------------------------------------

    for i, chunk in enumerate(chunks):

        chunk.metadata[
            "document_id"
        ] = document_id

        chunk.metadata[
            "chunk_number"
        ] = i

    # -----------------------------------------------------
    # 5. Create vector IDs
    # -----------------------------------------------------

    ids = [
        f"{document_id}_{i}"
        for i in range(
            len(chunks)
        )
    ]

    # -----------------------------------------------------
    # 6. Store in Pinecone
    # -----------------------------------------------------

    count = store_documents(
        chunks,
        ids
    )

    logger.info(
        "Completed %s: %d vectors",
        key,
        count
    )

    return count