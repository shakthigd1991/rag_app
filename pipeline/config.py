# pipeline/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # .env is one level up

# Needed now (query + index path)
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "rag-index")
EMBED_MODEL      = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Needed later (Drive + S3) — optional for now
AWS_REGION      = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET       = os.getenv("S3_BUCKET")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")