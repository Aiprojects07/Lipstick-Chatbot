
#!/usr/bin/env python3
"""
Generalized product ingestion to Pinecone for three JSON formats:
- Q&A <product>.json
- Snapshot <product>.json
- Ultimate_Lipstick_Attribute_System_Indian.json (taxonomy/attribute system)

Usage:
  python ingest_pinecone.py
  (Paths are hardcoded to the three JSON files under the repository's data/ folder.)

Environment:
  PINECONE_API_KEY=...
  OPENAI_API_KEY=...
  # Optional defaults via env
  PINECONE_INDEX_NAME=products-general
  PINECONE_ENVIRONMENT=us-east-1
  PINECONE_NAMESPACE=default
  OPENAI_EMBEDDING_MODEL=text-embedding-3-large

Notes:
- Hybrid (sparse) support is stubbed; you can plug in a BM25/SPLADE generator and pass as 'sparse_values' if needed.
- This script derives product_id from filenames unless a product_id can be inferred from internal JSON keys.
"""

import os
import re
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Iterable, Optional
from dotenv import load_dotenv
import logging

# Load environment variables from a .env file if present
load_dotenv()

# ---------- Logging ----------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
logger.debug("Logging initialized with level %s", LOG_LEVEL)

# Feature flags (can be controlled via .env)
INCLUDE_DEBUG_METADATA = os.getenv("INCLUDE_DEBUG_METADATA", "0").lower() in ("1", "true", "yes")
INCLUDE_DOC_TYPE = os.getenv("INCLUDE_DOC_TYPE", "1").lower() in ("1", "true", "yes")

# Verification-first workflow flags
VERIFY_ONLY = os.getenv("VERIFY_ONLY", "0").lower() in ("1", "true", "yes")
PREVIEW_PATH = os.getenv("PREVIEW_PATH")  # if set, write preview JSON here
PREVIEW_SAMPLES_PER_TYPE = int(os.getenv("PREVIEW_SAMPLES_PER_TYPE", "5"))

# Include toggles per doc type
INCLUDE_QA = os.getenv("INCLUDE_QA", "1").lower() in ("1", "true", "yes")
INCLUDE_SNAPSHOT = os.getenv("INCLUDE_SNAPSHOT", "1").lower() in ("1", "true", "yes")
INCLUDE_ATTRIBUTES_DOC = os.getenv("INCLUDE_ATTRIBUTES", "1").lower() in ("1", "true", "yes")

# ---------- Embedding Providers ----------

class Embedder:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

def _batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-large"):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        # OpenAI API supports up to ~8192 inputs per minute; we batch for safety
        logger.info("Embedding %d texts using model '%s'", len(texts), self.model)
        vectors = []
        for chunk in _batched(texts, 100):
            logger.debug("Embedding batch of size %d", len(chunk))
            resp = self.client.embeddings.create(model=self.model, input=chunk)
            vectors.extend([d.embedding for d in resp.data])
        logger.info("Embedding complete: %d vectors produced", len(vectors))
        return vectors

def make_embedder(embedding_model: str) -> Embedder:
    return OpenAIEmbedder(model=embedding_model)

# ---------- Pinecone client ----------

def get_pinecone_index(index_name: str, environment: str):
    from pinecone import Pinecone, ServerlessSpec
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")

    pc = Pinecone(api_key=api_key)
    # Create index if missing (serverless)
    existing = {i["name"] for i in pc.list_indexes()}
    if index_name not in existing:
        logger.info("Creating Pinecone index '%s' in region '%s'", index_name, environment)
        pc.create_index(
            name=index_name,
            dimension=3072,  # text-embedding-3-large
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=environment),
        )
    else:
        logger.info("Using existing Pinecone index '%s'", index_name)
    return pc.Index(index_name)

# ---------- Utilities ----------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\s/]+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text

def detect_doc_type(obj: dict, file_name: str) -> str:
    # Heuristics for the three formats
    if any(k.endswith("_complete_qa") for k in obj.keys()):
        return "qa"
    if {"section_4", "section_5", "section_6", "section_7"}.issubset(set(obj.keys())):
        return "snapshot"
    if "classification" in obj and "performance_metrics" in obj:
        return "attributes"
    # fallback by filename hints
    low = file_name.lower()
    if "q&a" in low or "qa " in low:
        return "qa"
    if "snapshot" in low:
        return "snapshot"
    if "attribute" in low or "system" in low:
        return "attributes"
    return "unknown"

TOPIC_KEYWORDS = {
    "transfer": ["transfer", "kiss", "cup", "mug", "smudge"],
    "wear_time": ["wear", "longevity", "hours", "fading", "touch-up"],
    "skin_tone": ["nc", "undertone", "fair", "medium", "deep", "cool", "warm", "neutral"],
    "ingredients": ["ingredient", "inci", "paraben", "fragrance", "essential oil"],
    "finish": ["matte", "gloss", "sheer", "opacity", "coverage"],
    "price": ["price", "expensive", "value", "budget"],
    "comfort": ["dry", "hydrating", "tight", "balm"],
    "application": ["apply", "layer", "lip liner", "buildable", "feather"],
    "climate": ["humidity", "heat", "ac", "sweat"],
    "occasion": ["wedding", "office", "party", "daytime", "night"],
}

def infer_topics(text: str) -> List[str]:
    low = text.lower()
    topics = []
    for t, kws in TOPIC_KEYWORDS.items():
        if any(kw in low for kw in kws):
            topics.append(t)
    return topics or ["general"]

# ---------- Chunkers ----------

def chunk_qa(obj: dict, product_meta: Dict) -> Iterable[Dict]:
    # Find the root QA key like '<product_id>_complete_qa' (do not assume dict order)
    qa_keys = [k for k in obj.keys() if isinstance(k, str) and k.endswith("_complete_qa")]
    if not qa_keys:
        raise ValueError("Q&A JSON missing '<product>_complete_qa' root key")
    root_key = qa_keys[0]
    root_val = obj[root_key]
    product_id_from_key = root_key.replace("_complete_qa", "")
    product_id = product_meta.get("product_id") or product_id_from_key
    # Prefer explicit product_name from JSON if provided
    json_product_name = obj.get("product_name") if isinstance(obj, dict) else None
    # Enrich metadata from product_id pattern: brand_product_line_..._shade
    # Example: glossier_generation_g_jam -> brand=glossier, product_line='generation g', shade='jam'
    parts = product_id_from_key.split("_")
    if len(parts) >= 3:
        brand = parts[0]
        shade = parts[-1]
        product_line = " ".join(parts[1:-1])
        clean_meta = dict(product_meta)
        clean_meta["brand"] = brand.title()
        clean_meta["product_line"] = product_line
        clean_meta["shade"] = shade.title()
        # Human-friendly complete name: prefer JSON field if present; otherwise derive
        clean_meta["product_name"] = json_product_name or f"{brand.title()} {product_line.title()} {shade.title()}"
    else:
        clean_meta = dict(product_meta)
        if json_product_name:
            clean_meta["product_name"] = json_product_name
    for section, items in root_val.items():
        if not isinstance(items, list):
            continue
        for i, qa in enumerate(items, start=1):
            q = qa.get("Q", "").strip()
            a = qa.get("A", "").strip()
            why = qa.get("WHY", "")
            sol = qa.get("SOLUTION", "")
            text = f"Q: {q}\nA: {a}"
            if why:
                text += f"\nWHY: {why}"
            if sol:
                text += f"\nSOLUTION: {sol}"
            topics = list(set(infer_topics(q + " " + a)))
            yield {
                "text": text,
                "metadata": {
                    **clean_meta,
                    "product_id": product_id,
                    "doc_type": "qa",
                    "section": section,
                    "topic": topics,
                    "source_pointer": f"{section}[{i}]",
                }
            }

def chunk_snapshot(obj: dict, product_meta: Dict) -> Iterable[Dict]:
    title4 = obj.get("section_4", {}).get("title", "Pros/Cons/Who Should Buy")
    # include section 4 title
    if title4:
        yield {
            "text": f"Title: {title4}",
            "metadata": {
                **product_meta,
                "doc_type": "snapshot",
                "section": "section_4_title",
                "topic": ["title"],
                "source_pointer": "section_4.title",
            }
        }
    fmt = obj.get("section_4", {}).get("format", {})
    if fmt:
        for bucket, items in fmt.items():
            for i, line in enumerate(items, start=1):
                text = f"{bucket}: {line}"
                yield {
                    "text": text,
                    "metadata": {
                        **product_meta,
                        "doc_type": "snapshot",
                        "section": "section_4_" + slugify(bucket),
                        "topic": infer_topics(line),
                        "source_pointer": f"section_4.{bucket}[{i}]",
                    }
                }
    # include section 5 title and content
    title5 = obj.get("section_5", {}).get("title")
    if title5:
        yield {
            "text": f"Title: {title5}",
            "metadata": {
                **product_meta,
                "doc_type": "snapshot",
                "section": "section_5_title",
                "topic": ["title"],
                "source_pointer": "section_5.title",
            }
        }
    content = obj.get("section_5", {}).get("content")
    if content:
        yield {
            "text": f"Summary: {content}",
            "metadata": {
                **product_meta,
                "doc_type": "snapshot",
                "section": "section_5_summary",
                "topic": infer_topics(content),
                "source_pointer": "section_5.content",
            }
        }
    # include section 6 title and snapshot map
    title6 = obj.get("section_6", {}).get("title")
    if title6:
        yield {
            "text": f"Title: {title6}",
            "metadata": {
                **product_meta,
                "doc_type": "snapshot",
                "section": "section_6_title",
                "topic": ["title"],
                "source_pointer": "section_6.title",
            }
        }
    snap = obj.get("section_6", {}).get("snapshot", {})
    for key, val in snap.items():
        text = f"{key}: {val}"
        yield {
            "text": text,
            "metadata": {
                **product_meta,
                "doc_type": "snapshot",
                "section": "section_6_snapshot",
                "topic": infer_topics(f"{key} {val}"),
                "source_pointer": f"section_6.snapshot.{key}",
            }
        }
    # include section 7 title and sources
    title7 = obj.get("section_7", {}).get("title")
    if title7:
        yield {
            "text": f"Title: {title7}",
            "metadata": {
                **product_meta,
                "doc_type": "snapshot",
                "section": "section_7_title",
                "topic": ["title"],
                "source_pointer": "section_7.title",
            }
        }
    sources = obj.get("section_7", {}).get("sources", [])
    for i, src in enumerate(sources, start=1):
        yield {
            "text": f"Source: {src}",
            "metadata": {
                **product_meta,
                "doc_type": "snapshot",
                "section": "section_7_sources",
                "topic": ["sources"],
                "source_pointer": f"section_7.sources[{i}]",
            }
        }

def flatten_keys(d: dict, prefix="") -> List[str]:
    lines = []
    if isinstance(d, dict):
        for k, v in d.items():
            newp = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                lines.extend(flatten_keys(v, newp))
            else:
                lines.append(f"{newp}: {v}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            newp = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                lines.extend(flatten_keys(v, newp))
            else:
                lines.append(f"{newp}: {v}")
    return lines

def chunk_attributes(obj: dict, product_meta: Dict) -> Iterable[Dict]:
    # Create compact reference chunks so the LLM knows what fields exist
    for top, val in obj.items():
        if top in ("title", "note"):
            continue
        if isinstance(val, dict) or isinstance(val, list):
            # list leaf keys/values and chunk if very long (no cap to avoid data loss)
            leaves = flatten_keys(val)
            if not leaves:
                leaves = [str(val)]
            # chunk into parts of up to 100 lines to keep chunk size reasonable
            part_size = 100
            parts = [leaves[i:i+part_size] for i in range(0, len(leaves), part_size)]
            for idx, part in enumerate(parts, start=1):
                text = f"Attribute Group: {top} (part {idx}/{len(parts)})\n" + "\n".join(part)
                meta = {
                    "product_id": product_meta.get("product_id"),
                    "product_name": product_meta.get("product_name"),
                    "brand": product_meta.get("brand"),
                    "group": top,
                    "topic": [top],
                }
                if INCLUDE_DOC_TYPE:
                    meta["doc_type"] = "attributes"
                if INCLUDE_DEBUG_METADATA:
                    meta["source_pointer"] = f"attributes.{top}.part_{idx}"
                    meta["source"] = product_meta.get("source")
                yield {
                    "text": text,
                    "metadata": meta,
                }
        else:
            text = f"Attribute Group: {top}\n{val}"
            meta = {
                "product_id": product_meta.get("product_id"),
                "product_name": product_meta.get("product_name"),
                "brand": product_meta.get("brand"),
                "group": top,
                "topic": [top],
            }
            if INCLUDE_DOC_TYPE:
                meta["doc_type"] = "attributes"
            if INCLUDE_DEBUG_METADATA:
                meta["source_pointer"] = f"attributes.{top}"
                meta["source"] = product_meta.get("source")
            yield {
                "text": text,
                "metadata": meta,
            }

# ---------- Product meta helpers ----------

def derive_product_from_filename(path: Path) -> Dict:
    # Try to infer product name/brand from filename like:
    # "Q&A Glossier Generation G Lipstick in Jam.json"
    name = path.stem
    # strip common prefixes
    name = re.sub(r"^(Q&A|QA|Snapshot)\s*", "", name, flags=re.I)
    product_name = name.strip()
    product_id = slugify(product_name)
    brand = None
    # very naive: first token might be brand
    if product_name:
        brand = product_name.split()[0]
    return {
        "product_id": product_id,
        "product_name": product_name,
        "brand": brand or "unknown",
    }

# ---------- Main ingestion ----------

def build_chunks_for_file(path: Path, region_focus: str = "IN", product_name_override: Optional[str] = None, product_id_override: Optional[str] = None) -> List[Dict]:
    logger.info("Reading JSON file: %s", path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    doc_type = detect_doc_type(obj, path.name)
    logger.info("Detected doc_type='%s' for file '%s'", doc_type, path.name)
    base_meta = derive_product_from_filename(path)
    base_meta.update({
        "category": "lipstick",
        "sub_category": "sheer-matte-bullet",
        "region_focus": region_focus,
        "language": "en",
        "source": path.name,
        "version": "v1",
        "created_at": now_iso(),
        "updated_at": now_iso(),
    })
    # If a global product_name override is provided, apply it
    if product_name_override:
        base_meta["product_name"] = product_name_override
        # Generalized brand rule: first word of product_name
        try:
            base_meta["brand"] = (product_name_override.strip().split()[0]) or base_meta.get("brand", "unknown")
        except Exception:
            pass
    if product_id_override:
        base_meta["product_id"] = product_id_override
    # default product alias to unified product_name if present, else product_id
    if base_meta.get("product_name"):
        base_meta["product"] = base_meta["product_name"]
    elif base_meta.get("product_id"):
        base_meta["product"] = base_meta["product_id"]
    # Log the resolved identity per file prior to chunking
    logger.info(
        "Identity | file='%s' doc_type='%s' product_id='%s' product_name='%s' brand='%s'",
        path.name,
        doc_type,
        base_meta.get("product_id"),
        base_meta.get("product_name"),
        base_meta.get("brand"),
    )
    chunks = []
    if doc_type == "qa" and INCLUDE_QA:
        chunks.extend(list(chunk_qa(obj, base_meta)))
    elif doc_type == "snapshot" and INCLUDE_SNAPSHOT:
        chunks.extend(list(chunk_snapshot(obj, base_meta)))
    elif doc_type == "attributes" and INCLUDE_ATTRIBUTES_DOC:
        # Attribute file is global; tie it to product_id inferred from sibling files if any.
        # Here we store as generic schema with product_name as 'GLOBAL_SCHEMA' for reuse.
        base_meta_schema = dict(base_meta)
        # No schema markers needed; attributes will share the unified product_id and brand
        # Always use the unified product_name from QA
        base_meta_schema["product_name"] = product_name_override
        # Generalized brand rule: derive from unified product_name as first token
        try:
            pname = base_meta_schema.get("product_name", "").strip()
            if pname:
                base_meta_schema["brand"] = pname.split()[0]
        except Exception:
            pass
        # For schema, align 'product' with the unified product_name if provided; else fallback to product_id
        base_meta_schema["product"] = base_meta_schema.get("product_name") 
        chunks.extend(list(chunk_attributes(obj, base_meta_schema)))
    else:
        # Fallback: flatten and ingest
        leaves = flatten_keys(obj)
        for i, line in enumerate(leaves, start=1):
            chunks.append({
                "text": line,
                "metadata": {
                    **base_meta,
                    "doc_type": "unknown",
                    "section": "flat",
                    "topic": ["general"],
                    "source_pointer": f"flat[{i}]",
                }
            })
    logger.info("Built %d chunks for file '%s'", len(chunks), path.name)
    return chunks

def upsert_chunks(index, embedder: Embedder, chunks: List[Dict], namespace: str = "default", batch_size: int = 100) -> List[str]:
    logger.info("Preparing to upsert %d chunks to namespace '%s'", len(chunks), namespace)
    texts = [c["text"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    ids = []
    for i, m in enumerate(metas):
        pid = m["product_id"]
        doc_type = m["doc_type"]
        sp = slugify(m.get("source_pointer", str(i)))
        ids.append(f"{pid}:{doc_type}:{sp}:{i}")
    vectors = embedder.embed(texts)
    # assemble Pinecone records
    records = [{
        "id": ids[i],
        "values": vectors[i],
        # include 'content' so downstream fetches can display the actual text
        "metadata": {**metas[i], "content": texts[i]},
        # "sparse_values": {...}  # plug in if you implement hybrid sparse later
    } for i in range(len(chunks))]

    # batch upserts
    for batch in _batched(records, batch_size):
        logger.debug("Upserting batch of %d records to namespace '%s'", len(batch), namespace)
        index.upsert(vectors=batch, namespace=namespace)
    logger.info("Upsert complete for %d records", len(records))
    return ids

def main():
    # Settings from environment (with sensible defaults)
    raw_index_name = os.getenv("PINECONE_INDEX_NAME", "Lipstick-chatbot")
    # Normalize index name: lowercase and only [a-z0-9-]
    index_name = re.sub(r"[^a-z0-9-]", "-", raw_index_name.lower())
    if index_name != raw_index_name:
        logger.warning("Normalized PINECONE_INDEX_NAME from '%s' to '%s' to meet Pinecone naming rules", raw_index_name, index_name)
    environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    namespace = os.getenv("PINECONE_NAMESPACE", "default")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    region_focus = os.getenv("REGION_FOCUS", "IN")

    # Hardcoded data files
    data_files = [
        Path("/Users/ptah/Documents/Lipstick_Chatbot/data/Ultimate_Lipstick_Attribute_System_Indian.json"),
        Path("/Users/ptah/Documents/Lipstick_Chatbot/data/Snapshot Glossier Generation G Lipstick in Jam.json"),
        Path("/Users/ptah/Documents/Lipstick_Chatbot/data/Q&A Glossier Generation G Lipstick in Jam.json"),
    ]

    # Validate file existence
    missing = [str(p) for p in data_files if not p.exists()]
    if missing:
        logger.error("Missing data files: %s", ", ".join(missing))
        raise SystemExit("Missing data files:\n" + "\n".join(missing))

    logger.info("Starting ingestion | index='%s' env='%s' namespace='%s' model='%s' region_focus='%s'", index_name, environment, namespace, embedding_model, region_focus)
    embedder = make_embedder(embedding_model)
    index = get_pinecone_index(index_name, environment)

    # Discover a single global product_name strictly from the QA JSON
    product_name_override = None
    product_id_override = None
    try:
        qa_path = None
        # First pass: pick the file whose detected doc_type is 'qa'
        for p in data_files:
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if detect_doc_type(obj, p.name) == "qa":
                    qa_path = p
                    break
            except Exception:
                continue
        # Fallback: if not found via detect, try filename heuristic containing 'Q&A'
        if qa_path is None:
            for p in data_files:
                if "Q&A" in p.name or "QA" in p.name:
                    qa_path = p
                    break
        if qa_path is not None:
            qa_obj = json.loads(qa_path.read_text(encoding="utf-8"))
            if isinstance(qa_obj, dict) and qa_obj.get("product_name"):
                product_name_override = qa_obj.get("product_name").strip()
                # Compute unified product_id from product_name once
                pname = product_name_override.lower()
                pname = re.sub(r"[\s/]+", "_", pname)
                pname = re.sub(r"[^a-z0-9_]+", "", pname)
                pname = re.sub(r"_+", "_", pname).strip("_")
                product_id_override = pname or None
                logger.info("Using unified product_name='%s' and product_id='%s' (from QA)", product_name_override, product_id_override)
            else:
                logger.warning("QA file found but product_name missing: %s", qa_path)
        else:
            logger.warning("QA file not found among data files; product_name/product_id will not be unified from QA")
    except Exception as e:
        logger.warning("Failed to probe product_name from QA: %s", e)

    all_chunks = []
    for p in data_files:
        try:
            file_chunks = build_chunks_for_file(
                p,
                region_focus=region_focus,
                product_name_override=product_name_override,
                product_id_override=product_id_override,
            )
            all_chunks.extend(file_chunks)
            logger.info("[OK] %s: %d chunks", p.name, len(file_chunks))
        except Exception as e:
            logger.exception("[ERR] %s: %s", p.name, e)
    logger.info("Total chunks to upsert: %d", len(all_chunks))

    # Verification-only mode: print a concise report and optionally write preview JSON, then exit
    if VERIFY_ONLY:
        from collections import Counter, defaultdict
        by_type = defaultdict(list)
        for c in all_chunks:
            by_type[c["metadata"].get("doc_type", "unknown")].append(c)
        counts = {k: len(v) for k, v in by_type.items()}
        logger.info("Verification report (no upsert): %s", counts)
        for k, arr in by_type.items():
            logger.info("Sample %d/%d for doc_type=%s:", min(PREVIEW_SAMPLES_PER_TYPE, len(arr)), len(arr), k)
            for c in arr[:PREVIEW_SAMPLES_PER_TYPE]:
                meta = c["metadata"]
                logger.info("- section/group=%s | product=%s | snippet=%.120s", meta.get("section") or meta.get("group"), meta.get("product_name"), c["text"].replace("\n", " ")[:120])
        if PREVIEW_PATH:
            try:
                import json as _json
                from pathlib import Path as _Path
                _Path(PREVIEW_PATH).write_text(_json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("Wrote preview JSON to %s", PREVIEW_PATH)
            except Exception as e:
                logger.warning("Failed to write preview JSON: %s", e)
        return

    if not all_chunks:
        logger.warning("No chunks to upsert. Exiting.")
        return

    ids = upsert_chunks(index, embedder, all_chunks, namespace=namespace)
    # Basic verification: fetch a few records to confirm presence
    try:
        sample_ids = ids[:5]
        if sample_ids:
            got = index.fetch(ids=sample_ids, namespace=namespace)
            vectors = getattr(got, "vectors", {}) or {}
            found = list(vectors.keys())
            logger.info("Verification: fetched %d/%d sample records", len(found), len(sample_ids))
            for k in found:
                meta = vectors.get(k, {}).get("metadata", {})
                logger.debug("Sample ID=%s | doc_type=%s | section=%s | group=%s", k, meta.get("doc_type"), meta.get("section"), meta.get("group"))
    except Exception as e:
        logger.warning("Verification fetch failed: %s", e)
    logger.info("Ingestion pipeline finished successfully.")

if __name__ == "__main__":
    main()
