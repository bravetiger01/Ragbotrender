# Configuration settings
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBQsXJuElW6iDIsadQp_3tJyjhjCmYIZ04")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# Data paths
DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "data.json")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHUNKS_PATH = os.path.join(PROCESSED_DIR, "chunks.json")
VECTORS_PATH = os.path.join(PROCESSED_DIR, "vectors.npy")

os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200

# Retrieval parameters
TOP_K = 5
SEMANTIC_WEIGHT = 0.7
RERANKING_ENABLED = True

# Server settings
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

# Embedding model
EMBEDDING_MODEL = "all-mpnet-base-v2"