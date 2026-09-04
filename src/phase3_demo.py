"""
Phase 3 Demo — End-to-end ingestion and retrieval.
Usage: python src/phase3_demo.py data/sample_ml.txt "What is deep learning?"
"""

import sys
from pathlib import Path

# Add src/ to the path so we can import our own modules
sys.path.insert(0, str(Path(__file__).parent))

from loader import load_document
from chunker import chunk_pages
from embedder import embed_chunks, embed_query
from vector_store import index_chunks, query_database, get_collection

def main():
    if len(sys.argv) < 3:
        print("Usage: python src/phase3_demo.py <path_to_file> <\"your question\">")
        sys.exit(1)

    file_path = sys.argv[1]
    question = sys.argv[2]

    # ── Step 1: Ingest into Vector Store ─────────────────────────────────────────
    print(f"\n[1/3] Processing document: {file_path}")
    pages = load_document(file_path)
    chunks = chunk_pages(pages)
    
    print(f"[2/3] Embedding {len(chunks)} chunk(s)...")
    # This adds the 'embedding' key to each chunk dict
    chunks_with_embeddings = embed_chunks(chunks)
    
    print(f"[3/3] Indexing into ChromaDB...")
    index_chunks(chunks_with_embeddings)
    
    collection = get_collection()
    print(f"      Total chunks in database: {collection.count()}\n")

    # ── Step 2: Query the Vector Store ──────────────────────────────────────────
    print("=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)
    
    print("\nEmbedding question...")
    q_vec = embed_query(question)
    
    print("Searching ChromaDB for top 3 matches...\n")
    # n_results is the "k" in "top-k retrieval"
    results = query_database(q_vec, n_results=3)
    
    if not results:
        print("No results found.")
        return
        
    for idx, res in enumerate(results, start=1):
        print(f"Match #{idx} (Distance: {res['distance']:.4f})")
        print(f"Source: {res['source']} | Page: {res['page']}")
        # Show a snippet of the matching text
        preview = res['text'][:200].replace('\n', ' ')
        print(f"Text: {preview}...\n")
        
    print("[Done] Semantic search complete.")

if __name__ == "__main__":
    main()
