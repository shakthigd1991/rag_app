import hashlib
import logging

import boto3

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from config import (
    S3_BUCKET,
    S3_PREFIX,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from pipeline.index import vector_store

from ingestion.state import (
    initialize_database,
    get_document,
    save_document,
)


logger = logging.getLogger(__name__)


# =========================================================
# S3
# =========================================================

s3_client = boto3.client(
    "s3"
)


# =========================================================
# Text splitter
# =========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# =========================================================
# Document ID
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
# List Markdown files
# =========================================================

def list_markdown_files():

    files = []

    paginator = s3_client.get_paginator(
        "list_objects_v2"
    )

    for page in paginator.paginate(
        Bucket=S3_BUCKET,
        Prefix=S3_PREFIX,
    ):

        for obj in page.get(
            "Contents",
            []
        ):

            key = obj["Key"]

            if not key.lower().endswith(
                ".md"
            ):
                continue

            files.append({
                "key": key,
                "etag": obj.get(
                    "ETag",
                    ""
                ).strip('"'),
            })

    return files


# =========================================================
# Load Markdown
# =========================================================

def load_markdown(
    key
):

    response = s3_client.get_object(
        Bucket=S3_BUCKET,
        Key=key,
    )

    content = response[
        "Body"
    ].read().decode(
        "utf-8"
    )

    return Document(
        page_content=content,
        metadata={
            "bucket": S3_BUCKET,
            "s3_key": key,
            "file_name": key.split("/")[-1],
            "source": (
                f"s3://{S3_BUCKET}/{key}"
            ),
        },
    )


# =========================================================
# Process one document
# =========================================================

def process_document(
    key,
    etag,
):

    logger.info(
        "Processing %s",
        key
    )

    document = load_markdown(
        key
    )

    chunks = splitter.split_documents(
        [document]
    )

    if not chunks:

        logger.warning(
            "No chunks created for %s",
            key
        )

        return 0

    document_id = create_document_id(
        S3_BUCKET,
        key
    )

    ids = []

    for i, chunk in enumerate(
        chunks
    ):

        chunk.metadata[
            "document_id"
        ] = document_id

        chunk.metadata[
            "chunk_number"
        ] = i

        ids.append(
            f"{document_id}_{i}"
        )

    # -----------------------------------------------------
    # Upsert to Pinecone
    # -----------------------------------------------------

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    # -----------------------------------------------------
    # Save state
    # -----------------------------------------------------

    save_document(
        s3_key=key,
        etag=etag,
        document_id=document_id,
        chunk_count=len(chunks),
    )

    return len(chunks)


# =========================================================
# Sync
# =========================================================

def sync_documents():

    initialize_database()

    files = list_markdown_files()

    processed_documents = 0
    processed_chunks = 0
    skipped_documents = 0
    failed_documents = []

    for file in files:

        key = file["key"]
        etag = file["etag"]

        existing = get_document(
            key
        )

        # -------------------------------------------------
        # Already processed and unchanged
        # -------------------------------------------------

        if (
            existing
            and existing[1] == etag
        ):

            skipped_documents += 1

            logger.info(
                "Skipping unchanged: %s",
                key
            )

            continue

        # -------------------------------------------------
        # New or changed
        # -------------------------------------------------

        try:

            chunks = process_document(
                key,
                etag,
            )

            processed_documents += 1
            processed_chunks += chunks

        except Exception as exc:

            logger.exception(
                "Failed processing %s",
                key
            )

            failed_documents.append({
                "key": key,
                "error": str(exc),
            })

    return {
        "documents": processed_documents,
        "chunks": processed_chunks,
        "skipped": skipped_documents,
        "failed": failed_documents,
        "total_files": len(files),
    }
