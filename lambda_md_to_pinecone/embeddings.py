from langchain_openai import (
    OpenAIEmbeddings
)

from config import (
    OPENAI_API_KEY,
    EMBED_MODEL
)


# =========================================================
# Embedding Model
# =========================================================

embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    api_key=OPENAI_API_KEY
)