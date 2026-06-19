import os
from dotenv import load_dotenv

load_dotenv()

# Switch this to "pinecone" or "chroma"
VECTOR_DB = "chroma"

# Embedding model — must stay consistent across ingest and search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Chroma settings
CHROMA_PATH = "./chroma_storage"
CHROMA_COLLECTION_NAME = "tech_docs"

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "tech-docs-hybrid"

# Chunking settings
CHUNK_SIZE = 1  # number of sentences per chunk (our docs are already 1 sentence per line)

# Search settings
TOP_K = 3
HYBRID_ALPHA = 0.5  # 0 = pure keyword, 1 = pure semantic