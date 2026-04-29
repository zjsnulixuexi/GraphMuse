# GraphMuse Copilot

**GraphMuse Copilot** is an end-to-end GraphRAG demo for resume showcase:

- LLM fine-tuned with LoRA generates Cypher
- Neo4j returns structured graph facts
- Local markdown notes are retrieved as narrative context
- Qwen synthesizes the final answer for user-facing chat

## Structure

- `backend/main.py`: FastAPI service (`/api/chat`, `/api/health`)
- `backend/requirements.txt`: Python dependencies
- `frontend/`: static chat UI

## Run

```bash
cd /root/llm/gfm_paper/app/backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

## Optional env vars

- `GRAPHMUSE_BASE_MODEL` (default `/root/llm/model/qwen`)
- `GRAPHMUSE_ADAPTER_PATH` (default `/root/llm/gfm_paper/outputs/qwen25_gfm_lora`)
- `GRAPHMUSE_MD_DIR` (default `/root/llm/gfm_paper/md`)
- `GRAPHMUSE_NEO4J_URI` / `GRAPHMUSE_NEO4J_USER` / `GRAPHMUSE_NEO4J_PASSWORD`

## Resume one-liner

Built **GraphMuse Copilot**, a full-stack GraphRAG system (FastAPI + Neo4j + Qwen LoRA) that translates natural language to Cypher, executes graph retrieval, fuses markdown RAG context, and returns grounded Chinese answers in a web chat interface.
