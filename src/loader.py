"""
Phase 1 — Document Loader
─────────────────────────
Goal: Given a PDF or .txt file, extract raw text page-by-page and print it.
This is the very first link in the RAG pipeline — garbage in, garbage out.
If text extraction fails or mangles the content here, every downstream step suffers.
"""

import sys
from pathlib import Path
from pypdf import PdfReader   # pypdf reads PDF bytes and hands us page-level text


# ── helpers ────────────────────────────────────────────────────────────────────

def load_pdf(file_path: Path) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Returns a list of dicts — one per page — so we keep the page number around.
    We'll need that later for citations ("this answer came from page 4").

    dict shape: {"source": filename, "page": int (1-indexed), "text": str}
    """
    reader = PdfReader(str(file_path))  # PdfReader accepts a file path or file object
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()  # pypdf's OCR-free text extraction

        # Some scanned PDFs return None or whitespace — we skip those
        if not text or not text.strip():
            print(f"  [warn] Page {page_num} had no extractable text — skipped "
                  f"(might be a scanned image)")
            continue

        pages.append({
            "source": file_path.name,  # just the filename, not full path
            "page": page_num,
            "text": text.strip(),
        })

    return pages


def load_txt(file_path: Path) -> list[dict]:
    """
    For plain text files we treat the whole file as a single 'page'.
    We still return the same dict shape so the rest of the pipeline
    doesn't need to know whether the input was PDF or TXT.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return [{"source": file_path.name, "page": 1, "text": text.strip()}]


def load_document(file_path: str | Path) -> list[dict]:
    """
    Dispatch to the right loader based on file extension.
    Raises ValueError for unsupported types — fail loudly rather than silently.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix == ".txt":
        return load_txt(path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Only .pdf and .txt are supported."
        )


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Usage: python src/loader.py data/my_document.pdf
    if len(sys.argv) < 2:
        print("Usage: python src/loader.py <path_to_pdf_or_txt>")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"\n[LOADING] {file_path}\n{'-' * 50}")

    pages = load_document(file_path)

    print(f"[OK] Extracted {len(pages)} page(s)\n")

    # Print a preview of each page (first 300 chars) so you can verify extraction
    for p in pages:
        preview = p["text"][:300].replace("\n", " ")   # collapse newlines for readability
        print(f"[Page {p['page']}] {preview}...")
        print()
