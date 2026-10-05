import sqlite3
from pathlib import Path
from datetime import datetime


DB_PATH = Path("ingestion.db")


# =========================================================
# Database
# =========================================================

def get_connection():

    return sqlite3.connect(
        DB_PATH
    )


# =========================================================
# Initialize
# =========================================================

def initialize_database():

    with get_connection() as conn:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                s3_key TEXT PRIMARY KEY,
                etag TEXT,
                document_id TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                processed_at TEXT NOT NULL
            )
            """
        )

        conn.commit()


# =========================================================
# Get existing document
# =========================================================

def get_document(s3_key):

    with get_connection() as conn:

        cursor = conn.execute(
            """
            SELECT
                s3_key,
                etag,
                document_id,
                chunk_count,
                processed_at
            FROM documents
            WHERE s3_key = ?
            """,
            (s3_key,)
        )

        return cursor.fetchone()


# =========================================================
# Save document
# =========================================================

def save_document(
    s3_key,
    etag,
    document_id,
    chunk_count
):

    with get_connection() as conn:

        conn.execute(
            """
            INSERT INTO documents (
                s3_key,
                etag,
                document_id,
                chunk_count,
                processed_at
            )
            VALUES (?, ?, ?, ?, ?)

            ON CONFLICT(s3_key)
            DO UPDATE SET
                etag = excluded.etag,
                document_id = excluded.document_id,
                chunk_count = excluded.chunk_count,
                processed_at = excluded.processed_at
            """,
            (
                s3_key,
                etag,
                document_id,
                chunk_count,
                datetime.utcnow().isoformat(),
            )
        )

        conn.commit()
