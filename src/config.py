import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") # For Local Ollama
    FAISS_INDEX_PATH = "embeddings/faiss_index"
    DOCUMENTS_JSON_PATH = "embeddings/documents.json"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Multilingual Embedding Model

    EMBEDDING_DEVICE = "cpu"  # Force CPU usage to avoid meta tensor issues
    EMBEDDING_MODEL_KWARGS = {'device': 'cpu'}
    EMBEDDING_ENCODE_KWARGS = {'normalize_embeddings': True}
