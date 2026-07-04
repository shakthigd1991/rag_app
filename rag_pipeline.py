# rag_pipeline.py

import os
import uuid
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------
# ----- Create RAG Storage Pipeline -----
# ---------------------------------------

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create or connect to index
index_name = "rag-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # depends on embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

def get_rag_pipeline(file_path):
    """Loads documents (PDF or TXT), splits them, and builds Pinecone vectorstore."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    documents = loader.load()

    '''
    # Testing the metadata from PyMuPDFLoader
    for doc in documents:
        print(f"Document metadata: {doc.metadata}")
    '''

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Create vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Clear existing data in the index
    index.delete(delete_all=True)

    # For creating the metadata, we can use a unique document ID for each upload
    document_id = str(uuid.uuid4())

    for i, chunk in enumerate(chunks):
        chunk.metadata["document_id"] = document_id
        chunk.metadata["file_name"] = os.path.basename(file_path)
        chunk.metadata["chunk_number"] = i

    # Add chunks to vector store
    for i, chunk in enumerate(chunks):
        vector_store.add_texts(
            texts=[chunk.page_content],
            metadatas=[chunk.metadata],
            ids=[f"{document_id}_{i}"]
        )

# ---------------------------------------
# ----- Create RAG Retrieval Pipeline -----
# ---------------------------------------


    # Create RAG pipeline
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    def rag_pipeline(query, chat_history):
        # Retrieve relevant chunks
        relevant_chunks = vector_store.similarity_search(query, k=5)
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

        docs = vector_store.similarity_search(query, k=5)

        # Create prompt with context and chat history
        messages = [HumanMessage(content=query)]
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        # Get answer from LLM
        answer = llm(messages + [HumanMessage(content=prompt)])
        
        # Update chat history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer.content})

        citations = []
        for doc in docs:
            citations.append({
                "file_name": doc.metadata["file_name"],
                "chunk_number": doc.metadata["chunk_number"],
                "page: ": doc.metadata["page"] +1
            })
        
        return {
            "answer": answer.content,
            "citations": citations
        }

    return rag_pipeline