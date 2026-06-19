from search import search


test_cases = [
    {
        "query": "how do I split documents into smaller pieces",
        "expect_keywords": ["splitter", "chunk"]
    },
    {
        "query": "how does the system remember past conversation",
        "expect_keywords": ["memory", "conversation"]
    },
    {
        "query": "how can a workflow pause for human approval",
        "expect_keywords": ["human-in-the-loop", "approval"]
    },
    {
        "query": "what handles control flow between steps",
        "expect_keywords": ["edges", "conditional"]
    },
    {
        "query": "how to recover a workflow after a crash",
        "expect_keywords": ["checkpoint", "resumed"]
    },
]

def is_relevant(doc_text, expect_keywords):
    doc_lower = doc_text.lower()
    return any(keyword.lower() in doc_lower for keyword in expect_keywords)

def evaluate():
    total_precision = 0
    
    for case in test_cases:
        query = case["query"]
        expect_keywords = case["expect_keywords"]
        
        results = search(query, top_k=3)
        docs = results['documents'][0]
        
        relevant_count = sum(1 for doc in docs if is_relevant(doc, expect_keywords))
        precision_at_3 = relevant_count / 3
        total_precision += precision_at_3
        
        print(f"Query: '{query}'")
        print(f"Precision@3: {precision_at_3:.2f} ({relevant_count}/3 relevant)")
        for i, doc in enumerate(docs):
            mark = "✓" if is_relevant(doc, expect_keywords) else "✗"
            print(f"  [{mark}] {doc}")
        print()
    
    avg_precision = total_precision / len(test_cases)
    print(f"=== Average Precision@3 across {len(test_cases)} queries: {avg_precision:.2f} ===")

if __name__ == "__main__":
    evaluate()