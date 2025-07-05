import os
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from .config import Config

# Code that loads the vectorstore index if it exists, otherwise return None
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    index_file_path = os.path.join(Config.FAISS_INDEX_PATH, "index.faiss")
    if os.path.exists(index_file_path):
        return FAISS.load_local(Config.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

# Code to Retrieve top k documents similar to the query using the vectorstore
def retrieve_documents(query: str, k: int = 4):
    vectorstore = get_vectorstore()
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=k)
        return docs
    return []

