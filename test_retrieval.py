from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

results = db.similarity_search("login lockout", k=3)              #Test retrieval before adding the LLM
for r in results:
    print(r.metadata, "->", r.page_content[:100])