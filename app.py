# app.py
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from pipeline.clean import load_documents
from pipeline.chunk import chunk_documents
from pipeline.index import upsert_chunks
from query import answer_question

st.set_page_config(page_title="RAG Chatbot", page_icon="📘", layout="wide")
st.title("📘 RAG Chatbot with Memory")
st.write("Upload a PDF or TXT and ask questions interactively.")

with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file:
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.session_state.get("last_uploaded_file") != uploaded_file.name:
        with st.spinner("Reading and indexing…"):
            docs = load_documents(file_path)
            chunks, doc_id = chunk_documents(docs, uploaded_file.name)
            upsert_chunks(chunks, doc_id)
        st.session_state.chat_history = []
        st.session_state.last_uploaded_file = uploaded_file.name

    st.success(f"File `{uploaded_file.name}` indexed. Ask away!")

    user_input = st.chat_input("Ask a question...")
    if user_input:
        result = answer_question(user_input, st.session_state.chat_history,
                                 file_name=st.session_state.last_uploaded_file)
        with st.expander("Sources"):
            for c in result["citations"]:
                st.caption(f'{c["file_name"]} — chunk {c["chunk_number"]}'
                           + (f', page {c["page"]}' if c["page"] else ""))
        st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
else:
    st.info("Upload a PDF or TXT file to start chatting.")