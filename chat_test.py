import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

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

if __name__ == "__main__":
    print(answer_question("What test cases cover login?"))