# query.py
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from pipeline.index import vector_store   # reuse the same index connection

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

def answer_question(query, chat_history):
    search_query = rewrite_query(query, chat_history)
    docs = vector_store.similarity_search(query, k=5)
    for d in docs:
        print("MATCH from:", d.metadata.get("file_name"))   # shows in your terminal
    context = "\n\n".join(d.page_content for d in docs)

    messages = []
    for msg in chat_history:
        role = HumanMessage if msg["role"] == "user" else AIMessage
        messages.append(role(content=msg["content"]))

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    answer = llm.invoke(messages + [HumanMessage(content=prompt)])

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer.content})

    citations = [{
        "file_name": d.metadata.get("file_name"),
        "chunk_number": d.metadata.get("chunk_number"),
        "page": (d.metadata.get("page") + 1) if "page" in d.metadata else None,
    } for d in docs]

    return {"answer": answer.content, "citations": citations}

def rewrite_query(query, chat_history):
    if not chat_history:
        return query
    recent = "\n".join(f'{m["role"]}: {m["content"]}' for m in chat_history[-4:])
    prompt = (f"Given this conversation:\n{recent}\n\n"
              f"Rewrite this follow-up as a standalone question with no pronouns:\n{query}")
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()