# =============================================================================
# Pharmacy IT Operations Agent — Vector DB (FAISS on local disk)
#
# Demo steps 1–3 live here:
#   Upload Document   → save_upload()   (file lands in data/uploads/)
#   Ingest to Vector  → ingest_file()   (extract → chunk → embed → FAISS)
#   Retrieve          → retrieve()      (top-k with a relevance threshold)
#
# Every chunk gets a stable chunk_id "<filename>#<n>". That ID is what the
# agent cites in grounded answers, and what the citation check validates.
# The index is saved to data/index/ after every change, so it survives an app
# restart (demo step 8).
# =============================================================================

import io
import os
import shutil

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent import db

UPLOAD_DIR = os.path.join(db.DATA_DIR, "uploads")
INDEX_DIR = os.path.join(db.DATA_DIR, "index")
SAMPLES_DIR = os.path.join(db.DATA_DIR, "samples")

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

EMBEDDING_MODEL = "text-embedding-3-small"

DEFAULT_MIN_SCORE = 0.45

_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)


class UploadRejected(ValueError):
    pass


def _embeddings(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)


# ---- Step 1: Upload --------------------------------------------------------------
def save_upload(filename: str, data: bytes) -> dict:
    """Validates type and size BEFORE anything is stored (the step-1 guard)."""
    name = os.path.basename(filename)
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadRejected(f"'{ext}' files are not accepted. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRejected(f"File is {len(data) // 1024} KB; the limit is {MAX_UPLOAD_BYTES // 1024} KB")
    if not data:
        raise UploadRejected("File is empty")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(os.path.join(UPLOAD_DIR, name), "wb") as f:
        f.write(data)
    db.record_document(name, len(data), "uploaded")
    return {"filename": name, "size_bytes": len(data)}


def list_samples() -> list[str]:
    if not os.path.isdir(SAMPLES_DIR):
        return []
    return sorted(f for f in os.listdir(SAMPLES_DIR) if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS)


def read_sample(name: str) -> bytes:
    with open(os.path.join(SAMPLES_DIR, os.path.basename(name)), "rb") as f:
        return f.read()


# ---- Step 2: Ingest --------------------------------------------------------------
def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return data.decode("utf-8", errors="replace")


def ingest_file(filename: str, api_key: str) -> dict:
    """Extract → chunk → embed → add to FAISS → save index → update SQL."""
    path = os.path.join(UPLOAD_DIR, os.path.basename(filename))
    with open(path, "rb") as f:
        data = f.read()

    text = extract_text(filename, data)
    if not text.strip():
        db.record_document(filename, len(data), "failed: no text layer", 0)
        raise UploadRejected("No text could be extracted (is it a scanned PDF?)")

    chunks = _splitter.create_documents([text], metadatas=[{"source": filename}])
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{filename}#{i}"

    # Re-ingesting the same file replaces its old chunks instead of duplicating them.
    store = load_store(api_key)
    if store is not None:
        old_ids = [doc_id for doc_id, doc in store.docstore._dict.items() if doc.metadata.get("source") == filename]
        if old_ids:
            store.delete(old_ids)
        store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    else:
        store = FAISS.from_documents(chunks, _embeddings(api_key), ids=[c.metadata["chunk_id"] for c in chunks])

    store.save_local(INDEX_DIR)
    db.record_document(filename, len(data), "ingested", len(chunks))
    return {
        "filename": filename,
        "chunks": len(chunks),
        "sample_chunk_id": chunks[0].metadata["chunk_id"],
        "sample_text": chunks[0].page_content[:400],
    }


def load_store(api_key: str):
    if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        return None
    return FAISS.load_local(INDEX_DIR, _embeddings(api_key), allow_dangerous_deserialization=True)


def vector_count() -> int:
    """Counts vectors straight from the saved index — no API key needed."""
    path = os.path.join(INDEX_DIR, "index.faiss")
    if not os.path.exists(path):
        return 0
    import faiss
    return faiss.read_index(path).ntotal


def clear_index():
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    db.clear_documents()


# ---- Step 3: Retrieve ------------------------------------------------------------
def retrieve(query: str, api_key: str, k: int = 4, min_score: float = DEFAULT_MIN_SCORE) -> dict:
    """Returns every top-k hit with its score, and which ones pass the threshold.
    Showing the rejected hits too is deliberate: it makes the threshold visible.

    Score = cosine similarity. OpenAI embeddings are unit length and FAISS's
    flat L2 index returns SQUARED distance, so cosine = 1 - d / 2. In practice
    a passage that answers the question scores about 0.55; off-topic text ~0."""
    store = load_store(api_key)
    if store is None:
        return {"passed": [], "rejected": [], "min_score": min_score}
    hits = store.similarity_search_with_score(query, k=k)
    rows = [
        {"chunk_id": d.metadata["chunk_id"], "source": d.metadata["source"], "text": d.page_content,
         "score": round(1 - float(dist) / 2, 3)}
        for d, dist in hits
    ]
    return {
        "passed": [r for r in rows if r["score"] >= min_score],
        "rejected": [r for r in rows if r["score"] < min_score],
        "min_score": min_score,
    }
