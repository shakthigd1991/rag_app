import urllib.parse
import logging

from processor import (
    process_markdown
)


# =========================================================
# Logging
# =========================================================

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# =========================================================
# Lambda Handler
# =========================================================

def lambda_handler(
    event,
    context
):

    logger.info(
        "Received event: %s",
        event
    )

    processed = 0

    for record in event.get(
        "Records",
        []
    ):

        # -------------------------------------------------
        # Bucket
        # -------------------------------------------------

        bucket_name = (
            record[
                "s3"
            ][
                "bucket"
            ][
                "name"
            ]
        )

        # -------------------------------------------------
        # Object key
        # -------------------------------------------------

        encoded_key = (
            record[
                "s3"
            ][
                "object"
            ][
                "key"
            ]
        )

        key = urllib.parse.unquote_plus(
            encoded_key
        )

        logger.info(
            "Received: s3://%s/%s",
            bucket_name,
            key
        )

        # -------------------------------------------------
        # Only Markdown
        # -------------------------------------------------

        if not key.lower().endswith(
            ".md"
        ):

            logger.info(
                "Skipping non-Markdown: %s",
                key
            )

            continue

        # -------------------------------------------------
        # Only extracted-text
        # -------------------------------------------------

        if not key.startswith(
            "extracted-text/"
        ):

            logger.info(
                "Skipping file outside "
                "extracted-text/: %s",
                key
            )

            continue

        # -------------------------------------------------
        # Process
        # -------------------------------------------------

        count = process_markdown(
            bucket_name,
            key
        )

        logger.info(
            "Successfully processed %s: %d vectors",
            key,
            count
        )

        processed += 1

    return {
        "statusCode": 200,
        "processed": processed
    }