import os
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated import
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

# 1️⃣ Read college data
with open("college_details.txt", "r", encoding="utf-8") as f:
    college_text = f.read()

# 2️⃣ Load Hugging Face embedding model
print("🧠 Loading Hugging Face embedding model...")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3️⃣ Create or load ChromaDB
print("📥 Creating or loading Chroma database...")
db = Chroma.from_texts([college_text], embedding, persist_directory="./college_db_hf")

# 4️⃣ Load your fine-tuned Ollama model
model = OllamaLLM(model="gemma3-finetuned")

# 5️⃣ Continuous chat loop
while True:
    question = input("\n🤔 Ask about your college (or type '/bye' to exit): ")

    if question.strip().lower() in ["/bye", "exit", "quit"]:
        print("👋 Goodbye! Have a great day!")
        break

    os.system('cls' if os.name == 'nt' else 'clear')  # optional for cleaner view
    print("🔍 Performing semantic search...\n")

    # Perform semantic search silently
    docs = db.similarity_search(question, k=3)

    if not docs:
        print("⚠️ No relevant information found in ChromaDB.")
        continue

    # Combine retrieved docs into context
    context = "\n".join([d.page_content for d in docs])

    # Create a structured prompt
    prompt = f"""You are an assistant answering questions about a college.
Use the following information to answer accurately and clearly.

{context}

Question: {question}
Answer:"""

    print("🧠 Thinking...\n")
    answer = model.invoke(prompt)

    print("💬 Answer:\n")
    print(answer)
