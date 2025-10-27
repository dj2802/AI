# langgraph_rag.py

from typing import Any, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# 1️⃣ Load college data
with open("college_details.txt", "r", encoding="utf-8") as f:
    college_text = f.read()

# 2️⃣ Initialize Hugging Face embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3️⃣ Setup or load ChromaDB
print("📥 Creating or loading Chroma database...")
db = Chroma.from_texts([college_text], embedding_model, persist_directory="./college_db")


# 4️⃣ Define nodes
def ensure_db_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("✅ ChromaDB ready with embeddings.")
    state["db_ready"] = True
    return state


def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    print(f"🔍 Performing semantic search for: {question}")
    docs = db.similarity_search(question)
    context = "\n".join([d.page_content for d in docs])
    state["context"] = context
    return state


def generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    model = OllamaLLM(model="gemma3-finetuned")
    question = state["question"]
    context = state["context"]

    prompt = f"Use the following college data to answer accurately:\n{context}\n\nQuestion: {question}"
    print("\n🧠 Thinking...\n")
    answer = model.invoke(prompt)

    print("💬 Answer:\n")
    print(answer)
    state["answer"] = answer
    return state


# 5️⃣ Build LangGraph
graph = StateGraph(dict)
graph.add_node("ensure_db", ensure_db_node)
graph.add_node("retrieval", retrieval_node)
graph.add_node("generator", generator_node)

graph.add_edge(START, "ensure_db")
graph.add_edge("ensure_db", "retrieval")
graph.add_edge("retrieval", "generator")
graph.add_edge("generator", END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# 6️⃣ Run the graph
if __name__ == "__main__":
    question = input("🤔 Ask about your college: ")
    initial_state = {"question": question}

    # ✅ Add this line — gives LangGraph a thread/session ID
    config = {"configurable": {"thread_id": "college_thread"}}

    # ✅ Pass config to the invoke method
    response = app.invoke(initial_state, config=config)

    print("\n===========================")
    print("✅ Final Answer:")
    print(response["answer"])
    print("===========================")
