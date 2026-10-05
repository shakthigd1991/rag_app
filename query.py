from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from pipeline.index import vector_store

from config import CHAT_MODEL


# =========================================================
# LLM
# =========================================================

llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0.2,
)


# =========================================================
# Answer question
# =========================================================

def answer_question(
    query,
    chat_history,
):

    # -----------------------------------------------------
    # Rewrite follow-up question
    # -----------------------------------------------------

    search_query = rewrite_query(
        query,
        chat_history,
    )

    # -----------------------------------------------------
    # Vector search
    # -----------------------------------------------------

    docs = vector_store.similarity_search(
        search_query,
        k=5,
    )

    # -----------------------------------------------------
    # Debug
    # -----------------------------------------------------

    for doc in docs:

        print(
            "MATCH:",
            doc.metadata.get(
                "file_name"
            )
        )

    # -----------------------------------------------------
    # Context
    # -----------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # -----------------------------------------------------
    # Conversation
    # -----------------------------------------------------

    messages = []

    for message in chat_history:

        if message["role"] == "user":

            messages.append(
                HumanMessage(
                    content=message[
                        "content"
                    ]
                )
            )

        elif message["role"] == "assistant":

            messages.append(
                AIMessage(
                    content=message[
                        "content"
                    ]
                )
            )

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = f"""
You are a helpful document assistant.

Use the provided context to answer the user's question.

Context:
{context}

Question:
{query}

Rules:
- Answer using the context whenever possible.
- Do not invent information.
- If the context does not contain enough information,
  say that you do not have enough information.
- Be concise but useful.
"""

    messages.append(
        HumanMessage(
            content=prompt
        )
    )

    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    response = llm.invoke(
        messages
    )

    # -----------------------------------------------------
    # Citations
    # -----------------------------------------------------

    citations = []

    for doc in docs:

        citations.append({
            "file_name": doc.metadata.get(
                "file_name"
            ),
            "chunk_number": doc.metadata.get(
                "chunk_number"
            ),
            "source": doc.metadata.get(
                "source"
            ),
        })

    return {
        "answer": response.content,
        "citations": citations,
        "search_query": search_query,
    }


# =========================================================
# Rewrite query
# =========================================================

def rewrite_query(
    query,
    chat_history,
):

    if not chat_history:

        return query

    recent = "\n".join(
        f'{message["role"]}: '
        f'{message["content"]}'
        for message in chat_history[-4:]
    )

    prompt = f"""
Conversation:

{recent}

Rewrite the following follow-up question
as a standalone search query.

Follow-up question:
{query}

Return only the rewritten search query.
"""

    response = llm.invoke(
        [
            HumanMessage(
                content=prompt
            )
        ]
    )

    return response.content.strip()
