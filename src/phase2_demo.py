"""
Phase 2 Demo — Run this to see chunking + embeddings in action.
Usage: python src/phase2_demo.py data/sample_ml.txt
"""

import sys
from pathlib import Path

# Add src/ to the path so we can import our own modules
sys.path.insert(0, str(Path(__file__).parent))

from loader import load_document
from chunker import chunk_pages, CHUNK_SIZE, CHUNK_OVERLAP
from embedder import embed_chunks

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/phase2_demo.py <path_to_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    # ── Step 1: Load ────────────────────────────────────────────────────────────
    print(f"\n[1/3] Loading document: {file_path}")
    pages = load_document(file_path)
    print(f"      Loaded {len(pages)} page(s)")

    # ── Step 2: Chunk ───────────────────────────────────────────────────────────
    print(f"\n[2/3] Chunking (size={CHUNK_SIZE} tokens, overlap={CHUNK_OVERLAP} tokens)...")
    chunks = chunk_pages(pages)
    print(f"      Created {len(chunks)} chunk(s)\n")

    # Print details of the first 3 chunks so you can inspect the split
    print("=" * 60)
    print("SAMPLE CHUNKS (first 3):")
    print("=" * 60)
    for chunk in chunks[:3]:
        print(f"\n  chunk_id   : {chunk['chunk_id']}")
        print(f"  source     : {chunk['source']}  |  page: {chunk['page']}")
        print(f"  token_count: {chunk['token_count']}")
        # Show a 150-char preview with newlines collapsed
        preview = chunk["text"][:150].replace("\n", " ")
        print(f"  text       : {preview!r}...")
    print()

    # ── Step 3: Embed ───────────────────────────────────────────────────────────
    print(f"[3/3] Embedding all {len(chunks)} chunk(s)...")
    chunks = embed_chunks(chunks)   # adds "embedding" key to every chunk dict

    # ── Show what an embedding actually looks like ───────────────────────────────
    print("\n" + "=" * 60)
    print("WHAT AN EMBEDDING LOOKS LIKE:")
    print("=" * 60)

    first_chunk = chunks[0]
    embedding   = first_chunk["embedding"]

    print(f"\n  Text (first 80 chars): {first_chunk['text'][:80]!r}")
    print(f"\n  Embedding dimension  : {len(embedding)}")
    print(f"  First 10 values      : {[round(v, 4) for v in embedding[:10]]}")
    print(f"  Min value            : {min(embedding):.4f}")
    print(f"  Max value            : {max(embedding):.4f}")

    # Demonstrate that similar texts produce closer vectors than unrelated texts.
    # We do this by embedding 3 hand-picked sentences directly, so the demo
    # works even when the document produces only 1 chunk.
    print("\n" + "=" * 60)
    print("SIMILARITY DEMO (cosine similarity via dot product):")
    print("  [1.0 = identical meaning | 0.0 = unrelated]")
    print("=" * 60)

    from embedder import embed_query  # single-sentence embedder

    def dot(a: list[float], b: list[float]) -> float:
        # Works as cosine similarity because embeddings are L2-normalized
        return sum(x * y for x, y in zip(a, b))

    s1 = "Machine learning helps computers learn from data."
    s2 = "AI systems can improve their performance through experience."  # similar meaning
    s3 = "The Eiffel Tower is located in Paris, France."                # unrelated

    e1 = embed_query(s1)
    e2 = embed_query(s2)
    e3 = embed_query(s3)

    print(f"\n  S1: {s1}")
    print(f"  S2: {s2}")
    print(f"  S3: {s3}")
    print(f"\n  sim(S1, S2) - similar meaning  : {dot(e1, e2):.4f}  <- should be HIGH")
    print(f"  sim(S1, S3) - unrelated meaning: {dot(e1, e3):.4f}  <- should be LOW")
    print(f"  sim(S1, S1) - identical        : {dot(e1, e1):.4f}  <- always 1.0")

    print(f"\n[Done] Pipeline: load -> chunk -> embed complete.")
    print(f"       Total chunks ready for vector store: {len(chunks)}\n")


if __name__ == "__main__":
    main()
