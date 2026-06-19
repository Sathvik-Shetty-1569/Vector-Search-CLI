import chromadb
from sentence_transformers import SentenceTransformer
import config

def load_documents(filepath):
    """Read documents.txt and split into chunks (one line = one chunk for now)."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines

def tag_topic(text):
    """Simple keyword-based topic tagging for metadata filtering."""
    text_lower = text.lower()
    if "langgraph" in text_lower or "node" in text_lower or "graph" in text_lower:
        return "langgraph"
    elif "rag" in text_lower or "retriev" in text_lower or "rerank" in text_lower:
        return "rag"
    else:
        return "langchain"

def ingest_chroma(chunks):
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)
    
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = model.encode(chunks).tolist()
    
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"topic": tag_topic(chunk), "text": chunk} for chunk in chunks]
    
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )
    print(f"Ingested {len(chunks)} chunks into Chroma collection '{config.CHROMA_COLLECTION_NAME}'")

if __name__ == "__main__":
    chunks = load_documents("data/documents.txt")
    print(f"Loaded {len(chunks)} chunks from documents.txt")
    
    if config.VECTOR_DB == "chroma":
        ingest_chroma(chunks)
    else:
        print("Pinecone ingestion not implemented yet — coming next")