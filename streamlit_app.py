#!/usr/bin/env python3
"""
Streamlit UI for the Lipstick Chatbot

- Uses the retrieval + Claude answering pipeline from core/search_and_answer.py
- Maintains conversation history per-browser session

Run:
  streamlit run streamlit_app.py

Environment (.env):
  PINECONE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
  PINECONE_INDEX_NAME, PINECONE_NAMESPACE, OPENAI_EMBEDDING_MODEL, CLAUDE_MODEL
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on sys.path so we can import core.search_and_answer
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _load_secrets_into_env():
    """Copy Streamlit secrets into os.environ so downstream modules using os.getenv work.
    Does not overwrite existing env vars if already set.
    """
    try:
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float)):
                os.environ.setdefault(str(k), str(v))
            # If nested sections exist, flatten top-level expected keys
            if isinstance(v, dict):
                for kk, vv in v.items():
                    os.environ.setdefault(str(kk), str(vv))
    except Exception:
        pass

# Load local .env for local dev
load_dotenv()
# Also map Streamlit Secrets -> env for Streamlit Cloud
_load_secrets_into_env()

# Import pipeline (generalized)
try:
    from core.sear_and_answer_general import (
        answer_with_claude_no_filters,
        conversation_memory,
        load_system_prompt,
    )
except Exception as e:
    st.error(f"Failed to import pipeline from core/sear_and_answer_general.py: {e}")
    st.stop()

# --------------- Helpers ---------------

def ensure_env_keys(keys: List[str]) -> List[str]:
    missing = []
    for k in keys:
        if not os.getenv(k):
            missing.append(k)
    return missing

# --------------- Sidebar ---------------

st.set_page_config(page_title="Lipstick Chatbot", page_icon="💄", layout="centered")

# (Sidebar settings removed as requested)

# --------------- Header ---------------

st.title("💄 Lipstick Chatbot")
st.caption("Broad semantic search + grounded answers (Pinecone + OpenAI + Claude)")

# --------------- Session State ---------------

if "chat" not in st.session_state:
    st.session_state.chat: List[Dict[str, Any]] = []

# --------------- Input Form ---------------

with st.form("chat_form", clear_on_submit=True):
    q = st.text_input("Ask a question", placeholder="e.g., Does it transfer on cups? NARS Dolce Vita")
    submitted = st.form_submit_button("Send")

# --------------- Validation ---------------

required_keys = [
    "PINECONE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_NAMESPACE",
]
missing = ensure_env_keys(required_keys)
if missing:
    st.error("Missing environment variables: " + ", ".join(missing))
    st.stop()

# --------------- Handle Submit ---------------

if submitted and q:
    with st.spinner("Thinking..."):
        try:
            out = answer_with_claude_no_filters(q)
            answer = out.get("answer", "")
            st.session_state.chat.append({"user": q, "bot": answer, "meta": out})
            # also record into the shared memory instance used by the pipeline
            conversation_memory.add_exchange(q, answer)
        except Exception as e:
            st.error(f"Error: {e}")

# --------------- Display Chat ---------------

for turn in st.session_state.chat:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["bot"]) 

# --------------- Footer removed (no extra details) ---------------
