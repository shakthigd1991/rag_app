import boto3
import os
import urllib.parse
import logging
import fitz


# =========================================================
# Logger
# =========================================================

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# =========================================================
# AWS S3 Client
# =========================================================

s3_client = boto3.client("s3")


# =========================================================
# Lambda Handler
# =========================================================

def lambda_handler(event, context):

    logger.info("Received event: %s", event)

    try:

        # -------------------------------------------------
        # Process every S3 record
        # -------------------------------------------------

        for record in event.get("Records", []):

            # ---------------------------------------------
            # Get bucket name
            # ---------------------------------------------

            bucket_name = (
                record["s3"]["bucket"]["name"]
            )

            # ---------------------------------------------
            # Get object key
            # ---------------------------------------------

            encoded_key = (
                record["s3"]["object"]["key"]
            )

            object_key = urllib.parse.unquote_plus(
                encoded_key
            )

            logger.info(
                "Processing file: s3://%s/%s",
                bucket_name,
                object_key
            )

            # ---------------------------------------------
            # Make sure it is a PDF
            # ---------------------------------------------

            if not object_key.lower().endswith(".pdf"):

                logger.info(
                    "Skipping non-PDF file: %s",
                    object_key
                )

                continue

            # ---------------------------------------------
            # Prevent processing files from output folder
            # ---------------------------------------------

            if object_key.startswith(
                "extracted-text/"
            ):

                logger.info(
                    "Skipping extracted file: %s",
                    object_key
                )

                continue

            # ---------------------------------------------
            # Process PDF
            # ---------------------------------------------

            output_key = process_pdf(
                bucket_name,
                object_key
            )

            logger.info(
                "Successfully created: %s",
                output_key
            )

        return {
            "statusCode": 200,
            "body": "PDF processing completed successfully"
        }

    except Exception as e:

        logger.exception(
            "PDF processing failed"
        )

        raise


# =========================================================
# Process PDF
# =========================================================

def process_pdf(bucket_name, object_key):

    # -----------------------------------------------------
    # Download PDF from S3
    # -----------------------------------------------------

    logger.info(
        "Downloading PDF from S3..."
    )

    response = s3_client.get_object(
        Bucket=bucket_name,
        Key=object_key
    )

    pdf_bytes = response["Body"].read()

    logger.info(
        "Downloaded PDF. Size: %d bytes",
        len(pdf_bytes)
    )

    # -----------------------------------------------------
    # Extract PDF text
    # -----------------------------------------------------

    extracted_text = extract_pdf_text(
        pdf_bytes
    )

    if not extracted_text.strip():

        logger.warning(
            "No text extracted from PDF: %s",
            object_key
        )

    # -----------------------------------------------------
    # Convert text to Markdown
    # -----------------------------------------------------

    markdown_content = convert_to_markdown(
        extracted_text,
        object_key
    )

    # -----------------------------------------------------
    # Create output key
    # -----------------------------------------------------

    output_key = create_output_key(
        object_key
    )

    # -----------------------------------------------------
    # Upload Markdown to S3
    # -----------------------------------------------------

    logger.info(
        "Uploading Markdown to: s3://%s/%s",
        bucket_name,
        output_key
    )

    s3_client.put_object(
        Bucket=bucket_name,
        Key=output_key,
        Body=markdown_content.encode("utf-8"),
        ContentType="text/markdown; charset=utf-8"
    )

    return output_key


# =========================================================
# Extract Text from PDF
# =========================================================

def extract_pdf_text(pdf_bytes):

    logger.info(
        "Opening PDF using PyMuPDF..."
    )

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    try:

        page_count = len(document)

        logger.info(
            "PDF contains %d pages",
            page_count
        )

        for page_number, page in enumerate(
            document,
            start=1
        ):

            logger.info(
                "Extracting page %d/%d",
                page_number,
                page_count
            )

            text = page.get_text("text")

            if text:

                pages.append(
                    f"<!-- Page {page_number} -->\n\n"
                    f"{text.strip()}"
                )

    finally:

        document.close()

    return "\n\n".join(pages)


# =========================================================
# Convert Extracted Text to Markdown
# =========================================================

def convert_to_markdown(
    text,
    original_key
):

    file_name = os.path.basename(
        original_key
    )

    title = os.path.splitext(
        file_name
    )[0]

    markdown = []

    # -----------------------------------------------------
    # Markdown title
    # -----------------------------------------------------

    markdown.append(
        f"# {title}"
    )

    markdown.append("")

    markdown.append(
        f"> Source: `{original_key}`"
    )

    markdown.append("")

    markdown.append("---")

    markdown.append("")

    # -----------------------------------------------------
    # Add extracted text
    # -----------------------------------------------------

    for line in text.splitlines():

        line = line.strip()

        if not line:

            markdown.append("")

            continue

        markdown.append(line)

        markdown.append("")

    return "\n".join(markdown)


# =========================================================
# Create Output S3 Key
# =========================================================

def create_output_key(object_key):

    file_name = os.path.basename(
        object_key
    )

    base_name = os.path.splitext(
        file_name
    )[0]

    output_key = (
        f"extracted-text/"
        f"{base_name}.md"
    )

    return output_key