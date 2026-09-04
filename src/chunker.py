"""
Phase 2 — Chunker
─────────────────
Goal: Split raw page text into overlapping fixed-size chunks.

WHY chunking at all?
  - LLMs have a context-window limit — we can't feed a whole 200-page PDF.
  - Embeddings work best on short, focused passages (not whole chapters).
  - Smaller chunks = more precise retrieval (you find the exact paragraph, not the chapter).

WHY token-based chunking (not character or sentence)?
  - LLM pricing and limits are measured in tokens, not characters.
  - The word "unimaginatively" is 1 token; "AI" is also 1 token — character count misleads.
  - We use the HuggingFace tokenizer for the BERT model family, which is the same
    tokenizer family as our embedding model (all-MiniLM-L6-v2).
  - ORIGINAL PLAN: tiktoken (OpenAI's tokenizer). It has no pre-built wheel for
    Python 3.14 yet and requires a Rust compiler to build from source — so we
    switched. For learning purposes the count difference between tokenizers is tiny.

WHY 50-token overlap?
  - If an important sentence falls exactly on a chunk boundary, it would be split.
  - Overlap ensures boundary sentences appear fully in at least one chunk.
  - Downside: we store redundant data (~12% extra). Acceptable at this scale.
"""

from transformers import AutoTokenizer

# bert-base-uncased uses a WordPiece vocabulary of ~30,000 tokens.
# all-MiniLM-L6-v2 is derived from this family, so token counts will
# be consistent between chunking and embedding.
# We load it once at module level — the weights are tiny (~250 KB).
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")

CHUNK_SIZE    = 400   # max tokens per chunk
CHUNK_OVERLAP = 50    # tokens shared between consecutive chunks


def chunk_text(text: str, source: str, page: int) -> list[dict]:
    """
    Split a single string into overlapping token-window chunks.

    Args:
        text:   The raw text to split (one page's worth).
        source: Filename — carried through for citations later.
        page:   Page number — carried through for citations later.

    Returns:
        List of chunk dicts, each with:
          {
            "chunk_id":    str  — unique ID like "sample_ml.txt_p1_c0"
            "source":      str  — filename
            "page":        int  — original page number
            "text":        str  — the chunk's raw text
            "token_count": int  — actual token count (≤ CHUNK_SIZE)
          }
    """
    # Step 1: tokenize the full page text → list of integer token IDs.
    # HuggingFace tokenizer adds special tokens ([CLS], [SEP]) by default;
    # add_special_tokens=False gives us a clean list of content tokens only.
    token_ids: list[int] = TOKENIZER.encode(text, add_special_tokens=False)
    # e.g. "Hello world" → [7592, 2088]  (WordPiece IDs, not BPE)

    chunks = []
    chunk_index = 0
    start = 0   # pointer into the token_ids list

    while start < len(token_ids):
        end = start + CHUNK_SIZE          # end of this window (exclusive)
        window = token_ids[start:end]     # slice out up to CHUNK_SIZE tokens

        # Step 2: decode the token IDs back to a human-readable string.
        # skip_special_tokens=True removes any [UNK]/[PAD] artifacts.
        chunk_text_str = TOKENIZER.decode(window, skip_special_tokens=True)

        chunks.append({
            # unique ID: we'll use this as the ChromaDB document ID in Phase 3
            "chunk_id":    f"{source}_p{page}_c{chunk_index}",
            "source":      source,
            "page":        page,
            "text":        chunk_text_str,
            "token_count": len(window),   # actual tokens (last chunk may be < 400)
        })

        chunk_index += 1

        # Step 3: advance the window, but step back by OVERLAP so chunks share tokens
        # If CHUNK_SIZE=400 and OVERLAP=50, next chunk starts at token 350
        step = CHUNK_SIZE - CHUNK_OVERLAP
        start += step

        # Edge case: if the remaining tokens are just the overlap tail, stop.
        # Without this we'd generate a tiny "stub" chunk of only overlap tokens.
        if start >= len(token_ids):
            break
        remaining = len(token_ids) - start
        if remaining <= CHUNK_OVERLAP:
            break

    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Convenience wrapper: chunk all pages from the loader's output.

    Args:
        pages: Output of loader.load_document() — list of page dicts.

    Returns:
        Flat list of all chunk dicts across all pages.
    """
    all_chunks = []
    for page in pages:
        page_chunks = chunk_text(
            text=page["text"],
            source=page["source"],
            page=page["page"],
        )
        all_chunks.extend(page_chunks)
    return all_chunks
