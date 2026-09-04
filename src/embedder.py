"""
Phase 2 — Embedder
──────────────────
Goal: Turn each text chunk into a fixed-size numeric vector (an "embedding").

WHAT is an embedding?
  An embedding is a list of floating-point numbers (a vector) that encodes
  the *meaning* of a piece of text. The key property:
    - Texts with similar meaning → vectors that are close together in space.
    - Texts with different meaning → vectors that are far apart.

  For example:
    "The cat sat on the mat"  → [0.21, -0.04, 0.87, ...]  (384 numbers)
    "A feline rested on a rug" → [0.20, -0.05, 0.85, ...]  (very close!)
    "Quantum entanglement"     → [-0.73, 0.61, -0.12, ...] (far away)

  This is how retrieval works in Phase 3: your *question* gets embedded too,
  and we find the chunks whose vectors are closest to the question vector.

WHY all-MiniLM-L6-v2?
  - 384-dimensional output (small = fast)
  - Trained specifically for semantic similarity tasks
  - Runs locally — no API call, no cost per embedding
  - The "L6" means 6 transformer layers; "Mini" means a distilled/smaller model

WHAT sentence-transformers is hiding:
  Under the hood it runs a transformer (BERT-family) model — tokenizes your text,
  passes it through 6 attention layers, then mean-pools the final hidden states
  into a single 384-d vector. The library handles all of this in one .encode() call.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model once at module import time (it's ~90 MB, don't reload repeatedly)
# First call downloads the model weights; subsequent calls use the local cache.
MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None   # lazy-loaded singleton


def get_model() -> SentenceTransformer:
    """Return the shared model instance, loading it on first call."""
    global _model
    if _model is None:
        print(f"[Embedder] Loading model '{MODEL_NAME}' (first run downloads ~90 MB)...")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[Embedder] Model loaded. Embedding dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an 'embedding' key to each chunk dict.

    We batch all texts in a single .encode() call — much faster than
    calling .encode() one chunk at a time because the GPU/CPU can
    process multiple texts in parallel.

    Args:
        chunks: List of chunk dicts from chunker.chunk_pages()

    Returns:
        The same list, each dict now has an extra key:
          "embedding": list[float]  — 384 floats, values roughly in [-1, 1]
    """
    model = get_model()

    # Extract just the text strings for the batch encode call
    texts = [chunk["text"] for chunk in chunks]

    # encode() returns a 2D numpy array of shape (num_chunks, 384)
    # show_progress_bar=True prints a tqdm bar — useful for large docs
    embeddings: np.ndarray = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,           # process 32 chunks at a time; tune if you OOM
        normalize_embeddings=True,  # L2-normalize so dot product == cosine similarity
    )
    # normalize_embeddings=True means: every vector has magnitude 1.
    # This lets us use dot product instead of cosine formula in Phase 3 — same result, faster.

    # Attach the embedding to each chunk dict
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()  # numpy → plain Python list for JSON-safety

    return chunks


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string — used in Phase 3 for similarity search.

    IMPORTANT: The query must be embedded with the SAME model as the chunks,
    otherwise the vector spaces don't align and similarity is meaningless.
    """
    model = get_model()
    vector: np.ndarray = model.encode(query, normalize_embeddings=True)
    return vector.tolist()
