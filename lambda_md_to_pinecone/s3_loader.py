import logging

import boto3

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


# =========================================================
# S3 Client
# =========================================================

s3_client = boto3.client("s3")


# =========================================================
# Load Markdown from S3
# =========================================================

def load_markdown_from_s3(
    bucket_name,
    key
):

    logger.info(
        "Reading s3://%s/%s",
        bucket_name,
        key
    )

    response = s3_client.get_object(
        Bucket=bucket_name,
        Key=key
    )

    content = response["Body"].read().decode(
        "utf-8"
    )

    document = Document(
        page_content=content,
        metadata={
            "bucket": bucket_name,
            "s3_key": key,
            "file_name": key.split("/")[-1],
            "source": (
                f"s3://{bucket_name}/{key}"
            )
        }
    )

    return document