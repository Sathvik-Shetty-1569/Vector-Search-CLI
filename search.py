import chromadb
from sentence_transformers import SentenceTransformer
import config

def search(query, topic_filter=None, top_k=config.TOP_K):
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)
    
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    query_embedding = model.encode(query).tolist()
    
    where_clause = {"topic": topic_filter} if topic_filter else None
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_clause
    )
    
    return results

def print_results(results, query):
    print(f"\n=== Query: '{query}' ===\n")
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    if not docs:
        print("No results found.")
        return
    
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
        print(f"Rank {i+1} | Score: {dist:.4f} | Topic: {meta['topic']}")
        print(f"  {doc}\n")

if __name__ == "__main__":
    print("=== Semantic Search CLI ===")
    print("Type 'exit' to quit. Type 'filter:topic' to set a topic filter (langchain/langgraph/rag). Type 'filter:none' to clear filter.\n")
    
    current_filter = None
    
    while True:
        query = input("Search query: ").strip()
        
        if query.lower() == "exit":
            break
        
        if query.lower().startswith("filter:"):
            topic = query.split(":", 1)[1].strip()
            current_filter = None if topic == "none" else topic
            print(f"Filter set to: {current_filter}\n")
            continue
        
        if not query:
            continue
        
        results = search(query, topic_filter=current_filter)
        print_results(results, query)