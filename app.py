import logging

import streamlit as st

from query import answer_question
from ingestion.ingest import sync_documents


# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO
)


# =========================================================
# Page
# =========================================================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📘",
    layout="wide",
)


st.title("📘 RAG Chatbot")

st.write(
    "Ask questions about the documents "
    "stored in the knowledge base."
)


# =========================================================
# Session state
# =========================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header(
        "Knowledge Base"
    )

    st.caption(
        "Synchronize new or changed "
        "Markdown files from S3."
    )

    sync_button = st.button(
        "🔄 Sync New Documents",
        type="primary",
        use_container_width=True,
    )

    if sync_button:

        with st.spinner(
            "Checking S3 and updating "
            "the knowledge base..."
        ):

            try:

                result = sync_documents()

                processed = result[
                    "documents"
                ]

                chunks = result[
                    "chunks"
                ]

                skipped = result[
                    "skipped"
                ]

                failed = result[
                    "failed"
                ]

                if failed:

                    st.warning(
                        f"Processed {processed} "
                        f"documents, created "
                        f"{chunks} chunks, "
                        f"but {len(failed)} "
                        f"documents failed."
                    )

                    with st.expander(
                        "View errors"
                    ):

                        for error in failed:

                            st.error(
                                f'{error["key"]}: '
                                f'{error["error"]}'
                            )

                elif processed == 0:

                    st.success(
                        "Knowledge base is already "
                        "up to date."
                    )

                    st.caption(
                        f"Skipped {skipped} "
                        f"unchanged documents."
                    )

                else:

                    st.success(
                        f"Sync complete! "
                        f"Processed {processed} "
                        f"documents and created "
                        f"{chunks} chunks."
                    )

                    st.caption(
                        f"Skipped {skipped} "
                        f"unchanged documents."
                    )

            except Exception as exc:

                st.error(
                    "Synchronization failed."
                )

                st.exception(
                    exc
                )


# =========================================================
# Chat input
# =========================================================

user_input = st.chat_input(
    "Ask a question..."
)


if user_input:

    # -----------------------------------------------------
    # Display user message
    # -----------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    with st.spinner(
        "Searching the knowledge base..."
    ):

        try:

            result = answer_question(
                user_input,
                st.session_state.chat_history[
                    :-1
                ],
            )

            answer = result[
                "answer"
            ]

        except Exception as exc:

            answer = (
                "Sorry, I couldn't process "
                "your question."
            )

            st.error(
                str(exc)
            )

            result = {
                "citations": []
            }

    # -----------------------------------------------------
    # Save assistant response
    # -----------------------------------------------------

    st.session_state.chat_history.append(
    {
        "role": "assistant",
        "content": answer,
        "citations": result.get(
            "citations",
            []
        ),
    }
)



# =========================================================
# Display chat history
# =========================================================

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

        citations = message.get(
            "citations",
            []
        )

        if citations:

            with st.expander(
                "📚 Sources"
            ):

                for citation in citations:

                    file_name = citation.get(
                        "file_name",
                        "Unknown"
                    )

                    chunk_number = citation.get(
                        "chunk_number",
                        ""
                    )

                    source = citation.get(
                        "source",
                        ""
                    )

                    st.caption(
                        f"**{file_name}** "
                        f"— chunk "
                        f"{chunk_number}"
                    )

                    if source:

                        st.caption(
                            source
                        )
