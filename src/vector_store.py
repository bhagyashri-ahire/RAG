"""
Phase 3 — Vector Store
──────────────────────
Goal: Store chunks and their embeddings, and query them by similarity.

WHAT is ChromaDB?
  It's a local vector database. We don't need a cloud service or a Docker container.
  It runs entirely in-process and saves its data to a folder (chroma_db/) on disk.

WHY use a vector database?
  Instead of looping through every chunk and calculating dot products one by one (like in the Phase 2 demo),
  a vector DB is optimized to do this across millions of vectors instantly using specialized indexes (like HNSW).

WHAT it hides from you:
  Chroma handles the math of distance calculations (Cosine, L2, Inner Product) and 
  maintains an index to avoid O(N) linear scans when querying.
"""

import chromadb
from pathlib import Path

# We'll save the database to a folder in the project root
DB_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "rag_documents"

def get_collection() -> chromadb.Collection:
    """
    Initialize the ChromaDB client and return the collection.
    If the collection doesn't exist, it creates it.
    """
    # PersistentClient saves data to disk so it survives between script runs
    client = chromadb.PersistentClient(path=str(DB_PATH))
    
    # We use cosine similarity (which is equivalent to dot product for normalized vectors)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def index_chunks(chunks: list[dict]) -> None:
    """
    Store chunks and their embeddings in ChromaDB.
    """
    collection = get_collection()
    
    # Chroma expects parallel lists for ids, embeddings, metadatas, and documents
    ids = []
    embeddings = []
    metadatas = []
    documents = []
    
    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "token_count": chunk["token_count"]
        })
    
    # Upsert means: insert if new, update if ID already exists
    # We batch them all in one call for efficiency
    if ids:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
def query_database(query_embedding: list[float], n_results: int = 3) -> list[dict]:
    """
    Find the top-K chunks most similar to the query embedding.
    """
    collection = get_collection()
    
    # We pass the pre-calculated query vector to Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    # Chroma returns lists of lists (because you can query multiple vectors at once).
    # We unpack the first result since we only passed one query vector.
    
    retrieved_chunks = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            retrieved_chunks.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "page": results["metadatas"][0][i]["page"],
                "distance": results["distances"][0][i],
            })
            
    return retrieved_chunks
