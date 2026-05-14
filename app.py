# app.py
import os
import streamlit as st
from dotenv import load_dotenv
# Load API keys
load_dotenv()
from rag_pipeline import get_rag_pipeline

# -----------------------------------
# UI of the Application
# -----------------------------------

st.set_page_config(page_title="RAG Chatbot", page_icon="📘", layout="wide")

st.title("📘 RAG Chatbot with Memory")
st.write("Upload a PDF or TXT and ask questions interactively.")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)
    file_path = os.path.join(data_folder, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Rebuild pipeline if file changes
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.rag = get_rag_pipeline(file_path)
        st.session_state.chat_history = []
        st.session_state.last_uploaded_file = uploaded_file.name


    # Initialize RAG pipeline in session_state
    if "rag" not in st.session_state:
        st.session_state.rag = get_rag_pipeline(file_path)
        st.session_state.chat_history = []

    st.success(f"File `{uploaded_file.name}` loaded successfully!")

    # Chat UI
    user_input = st.chat_input("Ask a question...")
    if user_input:
        result = st.session_state.rag(user_input, st.session_state.chat_history)
        st.session_state.last_answer = result["answer"]

    # Show conversation
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
else:
    st.info("Upload a PDF or TXT file to start chatting.")