import streamlit as st
import time
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Page Configuration ---
st.set_page_config(page_title="Department AI Chatbot", page_icon="🎓", layout="centered")

# --- Custom CSS for Modern UI ---
st.markdown("""
    <style>
    @keyframes gradientMove {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    body {
        background: linear-gradient(-45deg, #E3F2FD, #E8EAF6, #E0F7FA, #F3E5F5);
        background-size: 400% 400%;
        animation: gradientMove 15s ease infinite;
        font-family: 'Poppins', sans-serif;
    }
    .main { background: transparent; }
    .chat-box {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        max-height: 500px;
        overflow-y: auto;
        animation: fadeIn 1s ease;
    }
    .user-bubble {
        background-color: #C7D2FE;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        text-align: right;
        animation: fadeIn 0.7s ease;
    }
    .bot-bubble {
        background-color: #BBF7D0;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        text-align: left;
        animation: fadeIn 0.7s ease;
    }
    .header {
        text-align: center;
        font-weight: bold;
        font-size: 36px;
        margin-bottom: 10px;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeIn 1s ease;
    }
    .subtext {
        text-align: center;
        color: #555;
        font-size: 16px;
        margin-bottom: 30px;
        animation: fadeIn 1.2s ease;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        color: white;
        border-radius: 10px;
        font-weight: bold;
        transition: 0.3s;
        height: 3em;
        width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #06B6D4, #4F46E5);
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 class='header'>🎓 Department Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Ask anything about your department or student records!</p>", unsafe_allow_html=True)

# --- Load Resources Function ---
@st.cache_resource
def load_resources():
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "students.csv")
    college_path = os.path.join(base_dir, "college_details.txt")

    st.write("📂 Looking for CSV at:", csv_path)
    st.write("📂 Looking for college details at:", college_path)

    # --- Check CSV existence ---
    if not os.path.exists(csv_path):
        st.error(f"⚠️ File 'students.csv' not found! Expected path:\n{csv_path}")
        st.stop()

    # --- Read CSV ---
    df = pd.read_csv(csv_path)

    # --- Convert each student row into text (for better search) ---
    student_texts = []
    for _, row in df.iterrows():
        text_block = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        student_texts.append(text_block)

    # --- Read college details ---
    if not os.path.exists(college_path):
        st.error(f"⚠️ File 'college_details.txt' not found! Expected path:\n{college_path}")
        st.stop()

    with open(college_path, "r", encoding="utf-8") as f:
        college_text = f.read()

    # --- Create Embeddings + Vector DB ---
    embeddings = HuggingFaceEmbeddings(model_name="./local_hf_model")
    db = Chroma.from_texts(student_texts + [college_text], embeddings, persist_directory="./college_db")

    # --- Load Ollama Model ---
    model = OllamaLLM(model="gemma3-finetuned")

    st.success(f"✅ Loaded {len(df)} student records successfully!")
    return db, model


# --- Load Resources ---
db, model = load_resources()

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat ---
st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
for msg in st.session_state.messages:
    bubble_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
    icon = "🧑‍💻" if msg["role"] == "user" else "🤖"
    st.markdown(f"<div class='{bubble_class}'>{icon} {msg['content']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Input Section ---
st.markdown("### 💬 Ask your question:")
user_input = st.text_area("Type your question...", key="input", height=80)


def typewriter_effect(text):
    """Simulate typing animation."""
    placeholder = st.empty()
    output = ""
    for char in text:
        output += char
        placeholder.markdown(f"<div class='bot-bubble'>🤖 {output}</div>", unsafe_allow_html=True)
        time.sleep(0.02)
    return output


# --- Send Button Logic ---
if st.button("Send"):
    if user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})

        docs = db.similarity_search(user_input)
        context = "\n".join([d.page_content for d in docs])

        prompt = f"""
You are a department chatbot.

Answer the user's question ONLY using the data below.
Never explain, never include JSON, and never write any extra sentences.
Just give the final result as plain text or a simple table if multiple columns exist.

If the question is about a student, directly show that student's details in a readable format:
Example:
Name: Dhanush Raj
Email: dhanush.n.sns@...
Password: $CT@56789

If no match is found, say: "No record found in the dataset."

Data:
{context}

Question: {user_input}
"""

        def clean_output(answer):
            answer = answer.replace("```", "").replace("json", "").replace("Explanation:", "")
            return answer.strip()

        with st.spinner("🤔 Thinking..."):
            raw_answer = model.invoke(prompt)
            answer = clean_output(raw_answer)

        typewriter_effect(answer)
        st.session_state.messages.append({"role": "bot", "content": answer})
        st.rerun()
    else:
        st.warning("Please enter a question!")

# --- Footer ---
st.markdown(
    "<br><center><p style='color:gray;'>✨ Built by Vishnu using Streamlit + LangGraph + Ollama ✨</p></center>",
    unsafe_allow_html=True
)
