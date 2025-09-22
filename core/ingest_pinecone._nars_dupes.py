# In this file there is need to change only dupes json file path
#!/usr/bin/env python3
"""
Ingest a "dupe guide" JSON into Pinecone (full coverage, no data loss).

JSON schema expected (trimmed):
{
  "original_product": {
    "name": "NARS Air Matte Lip Color in Dolce Vita",
    "price_usd": 28,
    "shade_description": "Warm-toned ..."
  },
  "dupes": [
    {"rank": 1, "category": "HIGH-END", "brand": "MAC", "product": "Mehr",
     "shade": null, "price_usd": 22, "pros": [...], "cons": [...], "verdict": "..."},
     ...
  ],
  "best_picks": {...},
  "main_tradeoffs": {...},
  "shopping_strategy": {...},
  "notes": "..."
}

ENV to set before running:
  PINECONE_API_KEY=...
  OPENAI_API_KEY=...
  PINECONE_INDEX_NAME=products-general
  PINECONE_NAMESPACE=dupes
  PINECONE_ENVIRONMENT=us-east-1
  OPENAI_EMBEDDING_MODEL=text-embedding-3-large

Usage:
  python ingest_dupe_json_to_pinecone.py
"""

import os
import re
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
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
    return t

def stable_id(meta: Dict[str, Any], text: str) -> str:
    """Deterministic IDs so re-ingest overwrites the same content."""
    key = "|".join([
        str(meta.get("product_id","")),
        str(meta.get("doc_type","")),
        str(meta.get("section","")),
        str(meta.get("group","")),
        str(meta.get("rank","")),
        str(meta.get("dupe_brand","")),
        str(meta.get("dupe_product","")),
        str(meta.get("source","")),
        text
    ])
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return f"{slugify(meta.get('product_id','pid'))}:{slugify(meta.get('doc_type','doc'))}:{h}"

def parse_original_name(name: str) -> Dict[str, Optional[str]]:
    """
    Try to split "NARS Air Matte Lip Color in Dolce Vita" into:
    brand = NARS, product_line = Air Matte Lip Color, shade = Dolce Vita
    """
    brand, product_line, shade = None, None, None
    if not name:
        return {"brand": None, "product_line": None, "shade": None}
    # pattern like "... in <Shade>"
    m = re.search(r"\s+in\s+(.+)$", name, flags=re.I)
    if m:
        shade = m.group(1).strip()
        head = name[:m.start()].strip()
    else:
        shade = None
        head = name.strip()

    # first token as brand, rest as product_line
    parts = head.split()
    if parts:
        brand = parts[0]
        product_line = " ".join(parts[1:]) if len(parts) > 1 else None
    return {"brand": brand, "product_line": product_line, "shade": shade}

def join_nonempty(items: List[Optional[str]], sep: str = " | ") -> str:
    return sep.join([s for s in items if s])


# -------------------- pinecone + embeddings --------------------

def get_index(index_name: str, environment: str, dim: int):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing = {i["name"] for i in pc.list_indexes()}
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=environment),
        )
    return pc.Index(index_name)

def embed_texts(oai: OpenAI, model: str, texts: List[str], batch: int = 100) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), batch):
        part = texts[i:i+batch]
        resp = oai.embeddings.create(model=model, input=part)
        out.extend([d.embedding for d in resp.data])
    return out


# -------------------- chunk building --------------------

def build_chunks_from_dupe_json(data: Dict[str, Any], source_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Identity is read ONLY from this dupes JSON's top-level `product` block
    prod = (data or {}).get("product") or {}
    product_name_canonical = (prod.get("full_name") or "").strip()
    brand = (prod.get("brand") or "").strip()
    product_line = (prod.get("product_line") or "").strip()
    shade = (prod.get("shade") or "").strip()
    category = (prod.get("category") or "").strip()

    # Validate mandatory identity fields
    if not product_name_canonical:
        raise ValueError("dupes JSON must include top-level product.full_name for product identity.")
    if not shade:
        raise ValueError("dupes JSON must include top-level product.shade (Brand Product Line - Shade).")

    product_id = slugify(product_name_canonical)

    base_meta = {
        "product_id": product_id,
        "product_name": product_name_canonical,
        "brand": brand,
        "product_line": product_line,
        "shade": shade,
        "category": category,
        "doc_family": "dupe_guide",  # Changed to match report format pattern
        "language": "en",
    }

    records: List[Dict[str, Any]] = []

    # ---- 1) Original product chunk (from dupes JSON)
    op = data.get("original_product", {}) or {}
    op_name = (op.get("name") or "").strip()
    op_price = op.get("price_usd")
    op_shade_desc = op.get("shade_description")

    op_content_lines = [
        f"Original Product: {op_name}" if op_name else "Original Product",
        f"Price (USD): {op_price}" if op_price is not None else None,
        f"Shade description: {op_shade_desc}" if op_shade_desc else None,
        f"Category: {category}" if category else None,
    ]
    op_content = "\n".join([l for l in op_content_lines if l])

    meta = dict(base_meta)
    meta.update({
        "section": "original_product",
        "content": op_content,
        "price_usd": op_price,
        "shade_description": op_shade_desc,
        "is_original": True,
    })
    records.append({
        "id": stable_id(meta, op_content),
        "values": None,
        "metadata": meta
    })

    # ---- 2) Each dupe -> separate chunk
    dupes = data.get("dupes", []) or []
    for d in dupes:
        rank = d.get("rank")
        d_cat = d.get("category")
        d_brand = d.get("brand")
        d_product = d.get("product")
        d_shade = d.get("shade")
        d_price = d.get("price_usd")
        d_pros = d.get("pros") or []
        d_cons = d.get("cons") or []
        d_verdict = d.get("verdict")

        # content text (LLM-friendly)
        lines = [
            f"Dupe Rank: {rank}" if rank is not None else None,
            f"Category: {d_cat}" if d_cat else None,
            f"Brand: {d_brand}" if d_brand else None,
            f"Product: {d_product}" if d_product else None,
            f"Shade: {d_shade}" if d_shade else None,
            f"Price (USD): {d_price}" if d_price is not None else None,
            "Pros: " + "; ".join(d_pros) if d_pros else None,
            "Cons: " + "; ".join(d_cons) if d_cons else None,
            f"Verdict: {d_verdict}" if d_verdict else None,
        ]
        content = "\n".join([l for l in lines if l])

        meta = dict(base_meta)
        meta.update({
            "section": "dupes",
            "group": d_cat or "",
            "rank": rank if rank is not None else 0,
            "dupe_brand": d_brand or "",
            "dupe_product": d_product or "",
            "dupe_shade": d_shade or "",
            "price_usd_dupe": d_price if d_price is not None else 0,
            "pros": d_pros or [],
            "cons": d_cons or [],
            "verdict": d_verdict or "",
            "content": content,
        })
        records.append({
            "id": stable_id(meta, content),
            "values": None,
            "metadata": meta
        })

    # ---- 3) best_picks
    best = data.get("best_picks", {}) or {}
    if best:
        content = "\n".join([f"{k}: {v}" for k, v in best.items()])
        meta = dict(base_meta)
        meta.update({
            "section": "best_picks",
            "content": content,
        })
        records.append({
            "id": stable_id(meta, content),
            "values": None,
            "metadata": meta
        })

    # ---- 4) main_tradeoffs
    tradeoffs = data.get("main_tradeoffs", {}) or {}
    if tradeoffs:
        content = "\n".join([f"{k}: {v}" for k, v in tradeoffs.items()])
        meta = dict(base_meta)
        meta.update({
            "section": "main_tradeoffs",
            "content": content,
        })
        records.append({
            "id": stable_id(meta, content),
            "values": None,
            "metadata": meta
        })

    # ---- 5) shopping_strategy
    strategy = data.get("shopping_strategy", {}) or {}
    if strategy:
        # one chunk per strategy group
        for group, items in strategy.items():
            content = f"{group}: " + ", ".join(items)
            meta = dict(base_meta)
            meta.update({
                "section": "shopping_strategy",
                "group": group,
                "content": content,
            })
            records.append({
                "id": stable_id(meta, content),
                "values": None,
                "metadata": meta
            })

    # ---- 6) notes
    notes = data.get("notes")
    if notes:
        meta = dict(base_meta)
        meta.update({
            "section": "notes",
            "content": str(notes),
        })
        records.append({
            "id": stable_id(meta, str(notes)),
            "values": None,
            "metadata": meta
        })

    # ---- 7) raw JSON catch-all (guarantee total coverage)
    raw_txt = json.dumps(data, ensure_ascii=False, indent=2)
    meta = dict(base_meta)
    meta.update({
        "section": "raw_json_full",
        "content": raw_txt,
    })
    records.append({
        "id": stable_id(meta, raw_txt),
        "values": None,
        "metadata": meta
    })

    return records, base_meta


# -------------------- main --------------------

def main():
    # Load environment variables from .env file
    load_dotenv(override=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ingest_dupes.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting NARS dupes ingestion process")
    
    # Configuration: single source path for dupes JSON
    default_dupes_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "nars-dolce-vita-dupes.json",
    )
    json_file_path = os.getenv("DUPES_JSON_PATH", default_dupes_path)
    index_name = os.getenv("PINECONE_INDEX_NAME", "nars-air-matte-lip-color-dolce-vita")
    namespace = os.getenv("PINECONE_NAMESPACE", "default")
    environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    logger.info(f"Configuration: index={index_name}, namespace={namespace}, environment={environment}")
    logger.info(f"Embedding model: {embedding_model}")
    logger.info(f"JSON file path: {json_file_path}")

    # env checks
    if "PINECONE_API_KEY" not in os.environ:
        logger.error("PINECONE_API_KEY not set")
        raise RuntimeError("PINECONE_API_KEY not set")
    if "OPENAI_API_KEY" not in os.environ:
        logger.error("OPENAI_API_KEY not set")
        raise RuntimeError("OPENAI_API_KEY not set")

    logger.info("Environment variables validated successfully")

    # load JSON from single configured path
    try:
        logger.info(f"Loading JSON data from {json_file_path}")
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("JSON data loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load JSON data: {e}")
        raise

    # build records
    try:
        logger.info("Building records from dupe JSON")
        records, base_meta = build_chunks_from_dupe_json(data, source_name=os.path.basename(json_file_path))
        # Identity check print (visual confirmation before upsert)
        logger.info("IDENTITY | product_name='%s' product_id='%s' brand='%s' line='%s' shade='%s'",
                    base_meta.get("product_name"), base_meta.get("product_id"),
                    base_meta.get("brand"), base_meta.get("product_line"), base_meta.get("shade"))
        logger.info(f"Built {len(records)} chunks for product_name='{base_meta['product_name']}' (product_id='{base_meta['product_id']}')")
        print(f"[info] Built {len(records)} chunks for product_name='{base_meta['product_name']}' (product_id='{base_meta['product_id']}').")
    except Exception as e:
        logger.error(f"Failed to build records: {e}")
        raise

    # embed
    try:
        logger.info("Initializing OpenAI client and generating embeddings")
        oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        texts = [r["metadata"]["content"] for r in records]
        logger.info(f"Generating embeddings for {len(texts)} text chunks")
        vectors = embed_texts(oai, embedding_model, texts, batch=100)
        dim = len(vectors[0]) if vectors else 3072
        logger.info(f"Generated {len(vectors)} embeddings with dimension {dim}")
        for r, vec in zip(records, vectors):
            r["values"] = vec
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise

    # upsert
    try:
        logger.info(f"Connecting to Pinecone index '{index_name}'")
        index = get_index(index_name, environment, dim=dim)
        logger.info("Starting upsert process")
        for i in range(0, len(records), 100):
            part = records[i:i+100]
            index.upsert(vectors=part, namespace=namespace)
            logger.info(f"Upserted batch {i//100 + 1}: {len(part)} records")
        logger.info(f"Upsert completed: {len(records)} total records")
        print(f"[ok] Upserted {len(records)} records to index='{index_name}', namespace='{namespace}'.")
    except Exception as e:
        logger.error(f"Failed to upsert records: {e}")
        raise

    # verify: pull a few by product_id + section
    try:
        logger.info("Verifying upserted data")
        probe_vec = vectors[0]
        res = index.query(
            namespace=namespace,
            vector=probe_vec,
            top_k=8,
            include_metadata=True,
            filter={"product_id": {"$eq": base_meta["product_id"]}}
        )
        matches = res.get("matches", []) or getattr(res, "matches", [])
        logger.info(f"Verification successful: Retrieved {len(matches)} matches for product_id='{base_meta['product_id']}'")
        print(f"[verify] Retrieved {len(matches)} matches for product_id='{base_meta['product_id']}'. Sample:")
        for m in matches[:5]:
            md = m.get("metadata", {})
            print("  -", m.get("id"), "|", md.get("section"), "|", md.get("group"), "|", md.get("dupe_brand"), md.get("dupe_product"))
        
        logger.info("NARS dupes ingestion process completed successfully")
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise

if __name__ == "__main__":
    main()
