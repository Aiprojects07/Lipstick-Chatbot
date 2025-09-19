import os
import json
import time
from typing import Dict, List, Iterable
from tqdm import tqdm
import re
import logging

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv

# Load environment from .env if present
load_dotenv()

# ---------- logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# --------- Helpers for product identity ---------
import re

def slugify(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"[^\w\s-]+", "", t)
    t = re.sub(r"[\s/]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t or "unknown"


def load_product_identity() -> Dict[str, str]:
    """Load product fields from a report JSON (preferred), else from env, else defaults."""
    # Default to the report JSON used elsewhere in the repo
    default_report_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "nars-dolce-vita-report.json",
    )
    report_path = os.getenv("PRODUCT_INFO_JSON_PATH", default_report_path)

    name = os.getenv("PRODUCT_NAME")
    brand = os.getenv("PRODUCT_BRAND")
    line = os.getenv("PRODUCT_LINE")
    shade = os.getenv("PRODUCT_SHADE")

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            rpt = json.load(f)
        prod = (rpt or {}).get("product") or {}
        # Prefer JSON fields if present
        name = prod.get("full_name") or name
        brand = prod.get("brand") or brand
        line = prod.get("product_line") or line
        shade = prod.get("shade") or shade
    except Exception:
        # If report missing/unreadable, rely on env/defaults
        pass

    # Sensible defaults if still missing
    name = name or "NARS Air Matte Lip Color - Dolce Vita"
    brand = brand or "NARS"
    line = line or "Air Matte Lip Color"
    shade = shade or "Dolce Vita"

    pid = os.getenv("PRODUCT_ID") or slugify(name)

    return {
        "product_id": pid,
        "product_name": name,
        "brand": brand,
        "product_line": line,
        "shade": shade,
    }

# Materialize product identity once
_PROD = load_product_identity()
PRODUCT_ID = _PROD["product_id"]
PRODUCT_NAME = _PROD["product_name"]
BRAND = _PROD["brand"]
PRODUCT_LINE = _PROD["product_line"]
SHADE = _PROD["shade"]
# --------------------------------------------------------

# --------- Config (from env with sane defaults) ---------
# Pinecone index config
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "lipstick-chatbot").lower().replace("_", "-")

# Embedding model config
MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")  # 3072-dim for default
# If you might change MODEL, update EMBED_DIM accordingly (map models->dims if needed)
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))

# Data source path (relative to repo by default)
JSON_PATH = os.getenv(
    "ATTRIBUTES_JSON_PATH",
    str(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "nars-dolce-vita-attributes.by-category.json"))
)

# Upsert behavior — align with report script (uses BATCH_SIZE, default 100)
# Honor UPSERT_BATCH_SIZE as a fallback for backward compatibility.
BATCH_SIZE = int(os.getenv("BATCH_SIZE", os.getenv("UPSERT_BATCH_SIZE", "100")))

# Namespace segregation (None means default behavior)
NAMESPACE = os.getenv("PINECONE_NAMESPACE") or None

# Pinecone serverless spec
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
# --------------------------------------------------------

def load_data(path: str) -> Dict[str, Dict[str, List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Expected a dict of {category: {attribute_name: [values...]}}")
    return data

def yield_records(data: Dict[str, Dict[str, List[str]]]) -> Iterable[Dict]:
    """
    Yields items with no data loss: one record per individual value occurrence.
    ID format: <slug(category)>|<slug(attribute_name)>|<idx>
    """
    def attr_slug(s: str) -> str:
        return "".join(c if c.isalnum() else "-" for c in s).strip("-").lower()[:100]

    for category, attrs in data.items():
        if not isinstance(attrs, dict):
            continue
        for attr_name, values in attrs.items():
            if values is None:
                continue
            # ensure list, preserve multiplicity and order
            if not isinstance(values, list):
                values = [values]
            for i, val in enumerate(values):
                # Normalize to strings, but also keep original in metadata
                val_str = "" if val is None else str(val)
                # Text to embed: include rich context so retrieval is strong
                text = f"Category: {category}\nAttribute: {attr_name}\nValue: {val_str}"
                # Stable, collision-resistant id
                _id = f"{attr_slug(category)}|{attr_slug(attr_name)}|{i}"
                metadata = {
                    "category": category,
                    "attribute_name": attr_name,
                    "value": val_str,
                    "position": i,                 # occurrence index to preserve multiplicity
                    "source": os.path.basename(JSON_PATH),
                    "schema": "category->attribute_name->values[]",
                    # product identity for retrieval alignment
                    "product_id": PRODUCT_ID,
                    "product_name": PRODUCT_NAME,
                    "brand": BRAND,
                    "product_line": PRODUCT_LINE,
                    "shade": SHADE,
                    # align with other docs family/type
                    "doc_family": "lip_attributes",
                    "doc_type": "attribute_value",
                }
                yield {"id": _id, "text": text, "metadata": metadata}

def ensure_index(pc: Pinecone, index_name: str, dimension: int):
    logger.info("Ensuring Pinecone index exists | name='%s' dim=%s", index_name, dimension)
    existing = {ix.name: ix for ix in pc.list_indexes()}
    if index_name in existing:
        logger.debug("Index '%s' already exists", index_name)
        return
    logger.info("Creating index '%s' (cosine, dim=%s)", index_name, dimension)
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    # wait for ready
    while True:
        desc = pc.describe_index(index_name)
        if desc.status["ready"]:
            logger.info("Index '%s' is ready", index_name)
            break
        time.sleep(1)

def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    logger.debug("Embedding %d texts with model='%s'", len(texts), MODEL)
    resp = client.embeddings.create(model=MODEL, input=texts)
    # Ensure ordering is preserved
    vecs = [d.embedding for d in resp.data]
    return vecs

def chunked(iterable: Iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    logger.info("Starting attributes ingestion")
    logger.info("Config | index='%s' namespace=%s model='%s' batch_size=%s", INDEX_NAME, NAMESPACE, MODEL, BATCH_SIZE)
    logger.info("Paths | JSON_PATH='%s'", JSON_PATH)

    # Clients
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not openai_key or not pinecone_key:
        raise RuntimeError("OPENAI_API_KEY and PINECONE_API_KEY must be set in the environment.")
    client = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)

    # Ensure index exists
    ensure_index(pc, INDEX_NAME, EMBED_DIM)
    index = pc.Index(INDEX_NAME)

    # Load data and stream records
    data = load_data(JSON_PATH)
    records_iter = list(yield_records(data))
    logger.info("Prepared %d vectors to upsert (no data loss)", len(records_iter))

    # Upsert in batches
    total_batches = (len(records_iter) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi, batch in enumerate(tqdm(chunked(records_iter, BATCH_SIZE), total=total_batches, desc="Upserting"), start=1):
        texts = [r["text"] for r in batch]
        embeddings = embed_texts(client, texts)
        vectors = [
            {
                "id": r["id"],
                "values": emb,
                "metadata": r["metadata"]
            }
            for r, emb in zip(batch, embeddings)
        ]
        # Pinecone v3 upsert
        index.upsert(vectors=vectors, namespace=NAMESPACE)
        logger.debug("Upserted batch %d/%d | batch_size=%d", bi, total_batches, len(vectors))

    logger.info("Done. Data upserted to Pinecone index='%s'", INDEX_NAME)

if __name__ == "__main__":
    main()
