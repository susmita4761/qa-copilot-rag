import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(page_title="QA Copilot", page_icon="🧪")
st.title("🧪 QA Copilot")
st.caption("Ask about test cases, bugs, or requirements.")

@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

db = load_db()
llm = load_llm()

def answer_question(question):
    results = db.similarity_search(question, k=4)
    context = "\n\n---\n\n".join(
        f"[{r.metadata.get('category')}] {r.page_content}" for r in results
    )
    system_prompt = f"""You are a QA assistant. Answer using ONLY this context.
If the answer isn't in the context, say so.

Context:
{context}"""
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ])
    if isinstance(response.content, list):
        return response.content[0]["text"]
    return response.content

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_input = st.chat_input("Ask a QA question...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    with st.spinner("Thinking..."):
        answer = answer_question(user_input)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)