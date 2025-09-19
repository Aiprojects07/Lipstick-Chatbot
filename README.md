# Lipstick Chatbot

Broad semantic retrieval + grounded answers using Pinecone, OpenAI embeddings, and Anthropic Claude. Includes a Streamlit UI.

## Project Structure

- `core/ingest_pinecone.py` — Ingests the three JSON data files into Pinecone.
- `core/search_and_answer.py` — Retrieval + answer pipeline (no metadata filters). Now includes simple conversation memory.
- `core/verify_embeddings.py` — Verifies presence of specific content in Pinecone (with serverless-friendly fallbacks).
- `streamlit_app.py` — Streamlit UI for interactive chat.
- `data/` — Input JSON files and system prompt text.
- `requirements.txt` — Python dependencies.

## Local Setup

1) Create and activate a Python 3.9+ virtualenv (optional):

```bash
python3 -m venv venv
source venv/bin/activate
```

2) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3) Create a `.env` with your keys/settings:

```dotenv
PINECONE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
PINECONE_INDEX_NAME=lipstick-chatbot
PINECONE_NAMESPACE=default
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
CLAUDE_MODEL=claude-3-5-haiku-20241022
```

4) Ingest data (optional if you already ingested):

```bash
python core/ingest_pinecone.py
```

5) Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Streamlit Community Cloud Deployment

This app reads credentials from Streamlit Secrets. We also map Secrets into environment variables inside `streamlit_app.py` so the existing pipeline works without changes.

Required secrets:

- `PINECONE_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_NAMESPACE`
- (optional) `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-large`)
- (optional) `CLAUDE_MODEL` (default: `claude-3-5-haiku-20241022`)

Example `secrets.toml` (do NOT commit real secrets):

```toml
PINECONE_API_KEY = "pcn_..."
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
PINECONE_INDEX_NAME = "lipstick-chatbot"
PINECONE_NAMESPACE = "default"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
CLAUDE_MODEL = "claude-3-5-haiku-20241022"
```

### Steps

1) Push this repo to GitHub (see below).
2) Go to https://share.streamlit.io/ and connect your GitHub.
3) Create a new app, choose your repository and branch, and set `streamlit_app.py` as the entrypoint.
4) In the app settings, open the "Secrets" section and paste your `secrets.toml` content.
5) Deploy. Streamlit will build from `requirements.txt` automatically.

## Push to GitHub

Initialize the repo and commit (this repo already contains a `.gitignore` that excludes `venv/` and `.env`).

```bash
git init
git add .
git commit -m "Initial commit: Lipstick Chatbot"
```

Create a repository on GitHub (via web UI), then link and push:

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

Alternatively, if you have the GitHub CLI:

```bash
gh repo create <your-repo> --public --source=. --remote=origin --push
```

## Notes

- The app uses unfiltered semantic vector search (`top_k=15`) and selects up to 6 diverse chunks.
- For Pinecone Serverless, `core/verify_embeddings.py` avoids unsupported metadata filtering in `describe_index_stats` and falls back to content queries.
- Conversation memory is in-process only (not persisted). If you want persistent memory, open an issue or extend `ConversationMemory` to save/load JSON.
