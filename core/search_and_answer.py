#!/usr/bin/env python3
"""
Broad retrieval + Claude answering (NO FILTERS).
Works with your latest ingestion (unified product_id, brand from product_name's first word).

Flow:
1) Embed the user query with the SAME embedding model you used for ingestion.
2) Single broad Pinecone search (no product_id/brand filters at all).
3) Lightweight re-scoring + selection of 5–8 best chunks (prefers Q&A/Snapshot over Attributes).
4) Claude generates a grounded answer and explicitly names the product it’s answering for.

ENV (set before running):
  PINECONE_API_KEY=...
  OPENAI_API_KEY=...
  ANTHROPIC_API_KEY=...

  PINECONE_INDEX_NAME=products-general
  PINECONE_NAMESPACE=default
  OPENAI_EMBEDDING_MODEL=text-embedding-3-large
  CLAUDE_MODEL=claude-4.1   # or set to a Claude 3.5 Sonnet ID if needed

Usage:
  python search_and_answer_claude_broad.py "Does it transfer on cups? Glossier Generation G Jam"
"""

import os
import re
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import Counter, defaultdict

# deps: pip install pinecone-client openai anthropic
from pinecone import Pinecone
from openai import OpenAI
import anthropic
from dotenv import load_dotenv


# -------------------- utils --------------------

def env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    v = os.getenv(name, default)
    if required and v is None:
        raise RuntimeError(f"Missing env var: {name}")
    return v

# Load .env early so env() can find keys when script is launched directly
load_dotenv()

# ---------- logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# -------------------- simple conversation memory --------------------

class ConversationMemory:
    """Manages conversation history for context-aware responses."""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history: List[Dict[str, Any]] = []

    def add_exchange(self, user_message: str, bot_response: str, timestamp: Optional[str] = None):
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        exchange = {"timestamp": timestamp, "user": user_message, "bot": bot_response}
        self.conversation_history.append(exchange)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def get_conversation_context(self, include_last_n: int = 5) -> str:
        if not self.conversation_history:
            return ""
        recent = self.conversation_history[-include_last_n:]
        parts = ["Previous conversation:"]
        for ex in recent:
            parts.append(f"User: {ex['user']}")
            parts.append(f"Assistant: {ex['bot']}")
        return "\n".join(parts)

    def clear_history(self):
        self.conversation_history = []

    def get_history_as_list(self) -> List[Dict[str, Any]]:
        return self.conversation_history.copy()

    def load_history(self, history: List[Dict[str, Any]]):
        self.conversation_history = history[-self.max_history:] if history else []


# Global instance
conversation_memory = ConversationMemory()

def get_matches(obj: Any) -> List[Dict[str, Any]]:
    # Pinecone may return dict or an object with .matches
    if isinstance(obj, dict):
        return obj.get("matches", []) or []
    return getattr(obj, "matches", []) or []

def safe_text(md: Dict[str, Any]) -> str:
    return (md.get("content") or md.get("text") or "")

def short_label(i: int, md: Dict[str, Any]) -> str:
    # Include product_name to help the LLM keep products straight
    pn = md.get("product_name") or md.get("product") or "?"
    # Handle both doc_type and doc_family from different ingestion scripts
    dt = md.get("doc_type") or md.get("doc_family", "doc")
    # Handle section from both dupes and report scripts
    sec = md.get("section") or md.get("section_title") or md.get("group") or "?"
    return f"[{i}|{dt}|{sec}|{pn}]"

def embed_query(oai: OpenAI, model: str, text: str) -> List[float]:
    resp = oai.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding


# -------------------- chunk selection (no filters) --------------------

def rescore(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # No re-weighting; return matches as-is
    return list(matches)

def select_contexts_no_filters(matches: List[Dict[str, Any]], k: int = 6) -> List[Dict[str, Any]]:
    """
    NO FILTERS. We just:
      - take the top Pinecone matches as-is (no re-weighting)
      - lightly diversify by product_id so the LLM sees coherent evidence
    """
    rescored = rescore(matches)[:24]  # keep a decent pool
    by_pid = defaultdict(list)
    for m in rescored:
        pid = (m.get("metadata") or {}).get("product_id") or ""
        by_pid[pid].append(m)

    # Pick the most represented product first (still NO FILTERS, just ordering)
    pid_order = sorted(by_pid.keys(), key=lambda p: sum((mm.get("score") or 0.0) for mm in by_pid[p]), reverse=True)

    picked: List[Dict[str, Any]] = []
    logger.debug("Selecting contexts from %d product_ids", len(pid_order))
    # round-robin across pids to keep some diversity, but bias to the top pid
    while len(picked) < k and pid_order:
        for pid in pid_order:
            bucket = by_pid[pid]
            if bucket and len(picked) < k:
                picked.append(bucket.pop(0))
                logger.debug("  - picked from pid=%s", pid)
        # drop empty buckets
        pid_order = [p for p in pid_order if by_pid[p]]

    return picked


# -------------------- prompt for Claude --------------------

def make_prompt_for_claude(user_q: str, contexts: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, m in enumerate(contexts, start=1):
        md = m.get("metadata") or {}
        label = short_label(i, md)
        content = safe_text(md)[:1800]
        # Include a tiny metadata header to help the model ground answers
        header = []
        # Check for fields from both dupes and report ingestion scripts
        for key in ("product_name", "brand", "product_line", "shade", "doc_type", "doc_family", "section", "section_title", "group", "dupe_brand", "dupe_product", "rank"):
            val = md.get(key)
            if val and str(val).strip():  # Only include non-empty values
                header.append(f"{key}: {val}")
        header_txt = " | ".join(header)
        blocks.append(f"{label}\n{header_txt}\n{content}")
    ctx = "\n\n".join(blocks)
    history = conversation_memory.get_conversation_context(include_last_n=5)
    history_block = f"{history}\n\n" if history else ""
    # The system prompt (style, rule-set) will come from an external file; here we only provide the user question, prior conversation, and context
    return f"{history_block}Question: {user_q}\n\nContext Chunks:\n{ctx}"

def load_system_prompt() -> str:
    """Load the system prompt from a text file specified by SYSTEM_PROMPT_PATH env.
    Defaults to data/Chatbot system message prompt.txt resolved relative to the project root. If missing, returns a safe fallback."""
    default_path = str((Path(__file__).resolve().parent.parent / "data" / "Chatbot system message prompt.txt"))
    path = env("SYSTEM_PROMPT_PATH", default_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return content
    except Exception:
        pass
    return (
        "You are a concise, fact-grounded assistant. Answer only from the provided context. "
        "Keep responses brief and do not fabricate information."
    )


# -------------------- main pipeline (NO FILTERS) --------------------

def answer_with_claude_no_filters(query: str) -> Dict[str, Any]:
    # Clients & config
    t0 = time.perf_counter()
    pc = Pinecone(api_key=env("PINECONE_API_KEY", required=True))
    raw_index_name = env("PINECONE_INDEX_NAME", required=True)
    # Normalize to match ingestion/verify behavior
    index_name = raw_index_name.lower().replace("_", "-")
    namespace = env("PINECONE_NAMESPACE", required=True)
    index = pc.Index(index_name)
    logger.info("Using Pinecone index='%s' namespace='%s'", index_name, namespace)

    oai = OpenAI(api_key=env("OPENAI_API_KEY", required=True))
    embed_model = env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    anthropic_client = anthropic.Anthropic(api_key=env("ANTHROPIC_API_KEY", required=True))
    claude_model = env("CLAUDE_MODEL", "claude-3-5-haiku-20241022")
    t_init = time.perf_counter()
    logger.info("Init complete | models embed='%s' claude='%s' | %.1f ms", embed_model, claude_model, (t_init - t0) * 1000)

    # 1) Embed query
    t1 = time.perf_counter()
    qvec = embed_query(oai, embed_model, query)
    t2 = time.perf_counter()
    logger.info("Embedded query | %.1f ms", (t2 - t1) * 1000)

    # 2) Broad search ONLY (no product_id / brand filters)
    t3 = time.perf_counter()
    res = index.query(
        namespace=namespace,
        vector=qvec,
        top_k=15,
        include_metadata=True
    )
    t4 = time.perf_counter()
    matches = get_matches(res)
    logger.info("Pinecone query | matches=%d | %.1f ms", len(matches), (t4 - t3) * 1000)

    if not matches:
        return {
            "answer": "I couldn’t find any relevant information in the index.",
            "used_contexts": 0
        }

    # 3) Pick contexts (still no filters)
    t5 = time.perf_counter()
    contexts = select_contexts_no_filters(matches, k=6)
    t6 = time.perf_counter()
    logger.info("Selected contexts | k=%d | %.1f ms", len(contexts), (t6 - t5) * 1000)

    # 4) Build prompt & call Claude
    prompt = make_prompt_for_claude(query, contexts)
    system_prompt = load_system_prompt()
    
    # Debug: Print the actual prompt being sent to Claude
    print("\n=== DEBUG: PROMPT SENT TO CLAUDE ===")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("=== END DEBUG ===\n")
    t7 = time.perf_counter()
    msg = anthropic_client.messages.create(
        model=claude_model,
        max_tokens=700,
        temperature=0.2,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    t8 = time.perf_counter()
    answer_text = msg.content[0].text if msg and msg.content else ""
    logger.info("Claude call | %.1f ms | total %.1f ms", (t8 - t7) * 1000, (t8 - t0) * 1000)

    # Debug info: show top product_id tallies seen in the context pool
    pid_counts = Counter([(m.get("metadata") or {}).get("product_id") for m in matches if (m.get("metadata") or {}).get("product_id")])

    return {
        "answer": answer_text,
        "used_contexts": len(contexts),
        "top_product_candidates": pid_counts.most_common(3)
    }


# -------------------- CLI --------------------

def main():
    # CLI: interactive loop, remembers conversation across turns
    if len(sys.argv) >= 2:
        # One-shot mode with provided text, still records memory
        q = " ".join(sys.argv[1:])
        out = answer_with_claude_no_filters(q)
        print(out["answer"])
        conversation_memory.add_exchange(q, out.get("answer", ""))
        if "top_product_candidates" in out:
            logger.debug("top_product_candidates: %s", out["top_product_candidates"])
        return

    print("Enter your question (type 'exit' to quit, 'clear' to clear history): ")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q.lower() == "clear":
            conversation_memory.clear_history()
            print("Conversation history cleared.")
            continue
        out = answer_with_claude_no_filters(q)
        print(out.get("answer", ""))
        conversation_memory.add_exchange(q, out.get("answer", ""))
        if "top_product_candidates" in out:
            logger.debug("top_product_candidates: %s", out["top_product_candidates"])

if __name__ == "__main__":
    main()
