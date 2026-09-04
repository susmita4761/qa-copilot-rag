import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path("data")

def load_documents():                #Build the ingestion script (this is the "R" in RAG)
    all_docs = []
    for subfolder in DATA_DIR.iterdir():
        if not subfolder.is_dir():
            continue
        loader = DirectoryLoader(str(subfolder), glob="**/*.md", loader_cls=TextLoader)
        docs = loader.load()
        for doc in docs:
            doc.metadata["category"] = subfolder.name
        all_docs.extend(docs)
    return all_docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)   #Split into chunks.
    return splitter.split_documents(documents)

def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  #Embed and store in Chroma:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )
    return vectorstore

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    build_vectorstore(chunks)
    print("Vector store built at ./chroma_db")