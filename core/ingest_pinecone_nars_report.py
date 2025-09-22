# In this file there is need to change only report json file path
#!/usr/bin/env python3
"""
Ingest a Markdown-derived Q&A report JSON into Pinecone (full coverage).

Target JSON shape (from your converter):
{
  "title": "...",
  "source_file": "nars-dolce-vita-report.md",
  "extracted_at_utc": "...",
  "product": {
    "brand": "NARS",
    "product_line": "Air Matte Lip Color",
    "shade": "Dolce Vita",
    "full_name": "NARS Air Matte Lip Color - Dolce Vita: Complete Q&A Diagnostic Report"
  },
  "sections": [
    {
      "title": "Section 1: ...",
      "qas": [
        {"q": "...", "a": "...", "why": "...", "solution": "..."},
        ...
      ]
    },
    {
      "title": "Bottom Line",
      "content": "..."
    },
    {
      "title": "Quick Reference Snapshot",
      "snapshot": { "Feature": "Value", ... }
    }
  ]
}

What this script does:
- Creates deterministic, idempotent Pinecone IDs
- Chunks:
    * one meta chunk for the product
    * one chunk per Section.content
    * one chunk per Section.snapshot (flattened)
    * one chunk per QA (q+a+why+solution)
    * one raw_json_full catch-all chunk
- Embeds with OpenAI (text-embedding-3-large by default)
- Auto-creates the Pinecone index (serverless) if missing

ENV (set before running):
  PINECONE_API_KEY=...
  OPENAI_API_KEY=...
  PINECONE_INDEX_NAME=products-general
  PINECONE_NAMESPACE=reports
  PINECONE_ENVIRONMENT=us-east-1
  OPENAI_EMBEDDING_MODEL=text-embedding-3-large
  BATCH_SIZE=100
  FILE_PATH=data/nars-dolce-vita-report-2.json

Usage:
  python ingest_report_json_to_pinecone.py
"""

import os
import re
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv

# pip install pinecone-client openai python-dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI


# -------------------- utilities --------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def slugify(t: str) -> str:
    t = (t or "").lower()
    # keep only a-z, 0-9, spaces and hyphens during normalization
    t = re.sub(r"[^a-z0-9\s-]+", "", t)
    # convert whitespace and slashes to single hyphen
    t = re.sub(r"[\s/]+", "-", t)
    # collapse repeated hyphens and trim
    t = re.sub(r"-+", "-", t).strip("-")
    return t or "unknown"

def stable_id(*parts: str) -> str:
    key = "|".join(parts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:28]

def ensure_str(x: Any) -> str:
    return "" if x is None else str(x)

def flatten_snapshot(snapshot: Dict[str, Any]) -> str:
    lines = []
    for k, v in snapshot.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


# -------------------- Pinecone + embeddings --------------------

def ensure_index(index_name: str, region: str, dimension: int) -> "Index":
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    names = {i["name"] for i in pc.list_indexes()}
    if index_name not in names:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region),
        )
    return pc.Index(index_name)

def split_large_text(text: str, max_tokens: int = 8000) -> List[str]:
    """Split large text into chunks that fit within token limits (rough approximation: 1 token ≈ 4 chars)"""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at a sentence or paragraph boundary
        chunk = text[start:end]
        last_period = chunk.rfind('.')
        last_newline = chunk.rfind('\n')
        
        if last_period > len(chunk) * 0.8:  # If period is in last 20%
            end = start + last_period + 1
        elif last_newline > len(chunk) * 0.8:  # If newline is in last 20%
            end = start + last_newline + 1
        
        chunks.append(text[start:end])
        start = end
    
    return chunks

def embed_texts(oai: OpenAI, model: str, texts: List[str], batch_size: int = 100) -> List[List[float]]:
    out: List[List[float]] = []
    
    # Embed texts as-is (text splitting is handled at record level)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        resp = oai.embeddings.create(model=model, input=chunk)
        out.extend([d.embedding for d in resp.data])
    
    return out


# -------------------- records builder --------------------

def build_records_from_report(data: Dict[str, Any], source_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Identity
    product = data.get("product") or {}
    # Fetch full canonical product name exclusively from report JSON
    product_name = product.get("full_name") or ""
    brand = product.get("brand") or ""
    product_line = product.get("product_line") or ""
    shade = product.get("shade") or ""
    category = product.get("category") or ""
    # Enforce mandatory fields
    if not product_name:
        raise ValueError("product.full_name is required in the report JSON to build product identity.")
    if not shade:
        raise ValueError("product.shade is required in the report JSON to build product identity (Brand Product Line - Shade).")
    # Slugified human-readable deterministic ID
    product_id = slugify(product_name)

    base_meta = {
        "product_id": product_id,
        "product_name": product_name,
        "brand": brand,
        "product_line": product_line,
        "shade": shade,
        "category": category,
        "doc_family": "lip_report",          # family name for this JSON type
        "language": "en",
    }

    records: List[Dict[str, Any]] = []

    # 0) Product meta chunk
    meta_txt = "\n".join([
        f"Product: {product_name}",
        f"Brand: {brand}" if brand else "",
        f"Line: {product_line}" if product_line else "",
        f"Shade: {shade}" if shade else "",
        f"Category: {category}" if category else "",
        f"Title: {data.get('title') or ''}",
    ]).strip()
    r_id = f"{product_id}:meta:{stable_id(product_id, 'meta', meta_txt)}"
    records.append({
        "id": r_id,
        "values": None,
        "metadata": {
            **base_meta,
            "doc_type": "meta",
            "content": meta_txt
        }
    })

    # 1) Sections
    sections = data.get("sections") or []
    for s_idx, sec in enumerate(sections):
        title = ensure_str(sec.get("title")).strip()
        # 1a) Section content
        if sec.get("content"):
            content_txt = f"{title}\n{ensure_str(sec['content']).strip()}"
            r_id = f"{product_id}:section:{stable_id(product_id, 'section', str(s_idx), content_txt)}"
            records.append({
                "id": r_id,
                "values": None,
                "metadata": {
                    **base_meta,
                    "doc_type": "section_content",
                    "section_title": title,
                    "section_index": s_idx,
                    "content": content_txt
                }
            })

        # 1d) Title-only section (no content, no snapshot, no QAs)
        has_content = bool(sec.get("content"))
        has_snapshot = bool(sec.get("snapshot"))
        has_qas = bool(sec.get("qas"))
        if (not has_content) and (not has_snapshot) and (not has_qas) and title:
            title_only_txt = title
            r_id = f"{product_id}:section_title_only:{stable_id(product_id, 'section_title_only', str(s_idx), title_only_txt)}"
            records.append({
                "id": r_id,
                "values": None,
                "metadata": {
                    **base_meta,
                    "doc_type": "section_title_only",
                    "section_title": title,
                    "section_index": s_idx,
                    "content": title_only_txt
                }
            })

        # 1b) Snapshot
        if sec.get("snapshot"):
            snap_txt = f"{title}\n" + flatten_snapshot(sec["snapshot"])
            r_id = f"{product_id}:snapshot:{stable_id(product_id, 'snapshot', str(s_idx), snap_txt)}"
            records.append({
                "id": r_id,
                "values": None,
                "metadata": {
                    **base_meta,
                    "doc_type": "snapshot",
                    "section_title": title,
                    "section_index": s_idx,
                    "content": snap_txt
                }
            })

        # 1c) QAs
        for q_idx, qa in enumerate(sec.get("qas") or []):
            q = ensure_str(qa.get("q")).strip()
            a = ensure_str(qa.get("a")).strip()
            why = ensure_str(qa.get("why")).strip()
            sol = ensure_str(qa.get("solution")).strip()

            parts = [f"Q: {q}"]
            if a:   parts.append(f"A: {a}")
            if why: parts.append(f"WHY: {why}")
            if sol: parts.append(f"SOLUTION: {sol}")
            qa_txt = f"{title}\n" + "\n".join(parts)

            r_id = f"{product_id}:qa:{stable_id(product_id, 'qa', str(s_idx), str(q_idx), q, a, why, sol)}"
            records.append({
                "id": r_id,
                "values": None,
                "metadata": {
                    **base_meta,
                    "doc_type": "qa",
                    "section_title": title,
                    "section_index": s_idx,
                    "qa_index": q_idx,
                    "q": q,
                    "a": a or "",
                    "why": why or "",
                    "solution": sol or "",
                    "content": qa_txt
                }
            })

    # 2) Raw JSON catch-all
    raw_txt = json.dumps(data, ensure_ascii=False, indent=2)
    r_id = f"{product_id}:raw:{stable_id(product_id, 'raw_json_full')}"
    records.append({
        "id": r_id,
        "values": None,
        "metadata": {
            **base_meta,
            "doc_type": "raw_json_full",
            "content": raw_txt
        }
    })

    return records, base_meta


# -------------------- main --------------------

def main():
    # Load environment variables from .env file with override
    load_dotenv(override=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ingest_reports.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting NARS report ingestion process")
    
    # Configuration from environment variables only
    file_path = os.getenv("FILE_PATH", "/home/sid/Lipstick_chatbot_generalise/data/nars-dolce-vita-report.json")
    index_name = os.getenv("PINECONE_INDEX_NAME", "nars-air-matte-lip-color-dolce-vita")
    namespace = os.getenv("PINECONE_NAMESPACE", "default")
    environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    batch_size = int(os.getenv("BATCH_SIZE", "100"))

    logger.info(f"Configuration: index={index_name}, namespace={namespace}, environment={environment}")
    logger.info(f"Embedding model: {embedding_model}, batch_size: {batch_size}")
    logger.info(f"File path: {file_path}")

    # env checks
    if "PINECONE_API_KEY" not in os.environ:
        logger.error("PINECONE_API_KEY not set")
        raise RuntimeError("PINECONE_API_KEY not set")
    if "OPENAI_API_KEY" not in os.environ:
        logger.error("OPENAI_API_KEY not set")
        raise RuntimeError("OPENAI_API_KEY not set")

    logger.info("Environment variables validated successfully")

    # load JSON
    try:
        logger.info(f"Loading JSON data from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("JSON data loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load JSON data: {e}")
        raise

    # build records
    try:
        logger.info("Building records from report JSON")
        records, base_meta = build_records_from_report(data, source_name=os.path.basename(file_path))
        logger.info(f"Built {len(records)} records for product_id='{base_meta['product_id']}' product_name='{base_meta['product_name']}'")
        # Identity check print (visual confirmation before upsert)
        logger.info("IDENTITY | product_name='%s' product_id='%s' brand='%s' line='%s' shade='%s'",
                    base_meta.get("product_name"), base_meta.get("product_id"),
                    base_meta.get("brand"), base_meta.get("product_line"), base_meta.get("shade"))
        print(f"[info] Built {len(records)} records for product_id='{base_meta['product_id']}' product_name='{base_meta['product_name']}'.")
    except Exception as e:
        logger.error(f"Failed to build records: {e}")
        raise

    # embed
    try:
        logger.info("Initializing OpenAI client and generating embeddings")
        oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        texts = [r["metadata"]["content"] for r in records]
        logger.info(f"Generating embeddings for {len(texts)} text chunks with batch size {batch_size}")
        
        # Handle text splitting - create new records for split chunks
        final_records = []
        text_idx = 0
        
        for record in records:
            original_text = record["metadata"]["content"]
            text_chunks = split_large_text(original_text)
            
            if len(text_chunks) == 1:
                # No splitting needed
                final_records.append(record)
            else:
                # Create separate records for each chunk
                for chunk_idx, chunk_text in enumerate(text_chunks):
                    chunk_record = record.copy()
                    chunk_record["metadata"] = record["metadata"].copy()
                    chunk_record["metadata"]["content"] = chunk_text
                    chunk_record["metadata"]["chunk_index"] = chunk_idx
                    chunk_record["metadata"]["total_chunks"] = len(text_chunks)
                    # Update ID to include chunk info
                    original_id = record["id"]
                    chunk_record["id"] = f"{original_id}_chunk_{chunk_idx}"
                    final_records.append(chunk_record)
        
        # Generate embeddings for all final texts
        final_texts = [r["metadata"]["content"] for r in final_records]
        vectors = embed_texts(oai, embedding_model, final_texts, batch_size=batch_size)
        
        if not vectors:
            logger.error("No vectors produced; check inputs")
            raise RuntimeError("No vectors produced; check inputs.")
        dim = len(vectors[0])
        logger.info(f"Generated {len(vectors)} embeddings with dimension {dim} for {len(final_records)} records")
        
        # Assign vectors to records
        for rec, vec in zip(final_records, vectors):
            rec["values"] = vec
            
        # Update records to final_records
        records = final_records
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise

    # upsert
    try:
        logger.info(f"Connecting to Pinecone index '{index_name}'")
        index = ensure_index(index_name, environment, dimension=dim)
        logger.info("Starting upsert process")
        total = 0
        for i in range(0, len(records), batch_size):
            part = records[i:i+batch_size]
            index.upsert(vectors=part, namespace=namespace)
            total += len(part)
            logger.info(f"Upserted batch {i//batch_size + 1}: {len(part)} records")
        logger.info(f"Upsert completed: {total} total records")
        print(f"[ok] Upserted {total} vectors to index='{index_name}', namespace='{namespace}'.")
    except Exception as e:
        logger.error(f"Failed to upsert records: {e}")
        raise

    # verify: pull some back by product_id
    try:
        logger.info("Verifying upserted data")
        probe = vectors[0]
        res = index.query(
            namespace=namespace,
            vector=probe,
            top_k=10,
            include_metadata=True,
            filter={"product_id": {"$eq": base_meta["product_id"]}}
        )
        matches = res.get("matches", []) or getattr(res, "matches", [])
        logger.info(f"Verification successful: Retrieved {len(matches)} matches for product_id='{base_meta['product_id']}'")
        print(f"[verify] Retrieved {len(matches)} matches for product_id='{base_meta['product_id']}'.")
        for m in matches[:5]:
            md = m.get("metadata", {})
            print("  -", m.get("id"), "|", md.get("doc_type"), "| section:", md.get("section_title"), "| qa_idx:", md.get("qa_index"))
        
        logger.info("NARS report ingestion process completed successfully")
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise

if __name__ == "__main__":
    main()
