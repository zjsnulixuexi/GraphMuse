#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from typing import Any

os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from neo4j import GraphDatabase
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


APP_NAME = "GraphMuse Copilot"

BASE_MODEL_PATH = os.getenv("GRAPHMUSE_BASE_MODEL", "/root/llm/model/qwen")
ADAPTER_PATH = os.getenv("GRAPHMUSE_ADAPTER_PATH", "/root/llm/gfm_paper/outputs/qwen25_gfm_lora")
MD_DIR = os.getenv("GRAPHMUSE_MD_DIR", "/root/llm/gfm_paper/md")
NEO4J_URI = os.getenv("GRAPHMUSE_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("GRAPHMUSE_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("GRAPHMUSE_NEO4J_PASSWORD", "123456zzz")


CYPHER_SYSTEM_PROMPT = (
    "You are a graph foundation model expert. "
    "Translate user query into one Neo4j Cypher statement. "
    "Output Cypher only, no explanation."
)

ANSWER_SYSTEM_PROMPT = """你是 GraphMuse Copilot。
你会看到用户问题、图查询结果和文档摘录。请用简体中文给出清晰答案。
规则：
1) 结构化事实以图查询结果为准，不得编造数值或实体。
2) 若用户问“讲了什么/方法/贡献”，优先综合文档摘录并明确是“摘要性说明”。
3) 若图为空且文档也不足，明确说明“当前信息不足”。
4) 禁止输出 Cypher 代码。"""

CHITCHAT_SYSTEM_PROMPT = """你是友好的学术助手。若用户是寒暄或无关问题，简短自然回复，不要提 Cypher/Neo4j。"""

GRAPH_HINT = re.compile(
    r"(论文|作者|模型|数据集|指标|概念|关系|贡献|方法|讲了什么|Graph|GFM|Cypher|Neo4j|"
    r"MDGPT|GraphGPT|RAG|MultiGPrompt|OFA|LLaGa|SAMGPT|GraphMORE)",
    re.I,
)

CHITCHAT_EXACT = {"你好", "您好", "hi", "hello", "thanks", "谢谢", "拜拜", "bye", "在吗"}

FORBIDDEN_CYPHER = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH\s+DELETE|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH|GRANT|REVOKE)\b",
    re.I,
)


class ChatRequest(BaseModel):
    message: str
    debug: bool = False


class ChatResponse(BaseModel):
    answer: str
    mode: str
    cypher: str | None = None
    row_count: int | None = None


def normalize_text(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"^User>\s*", "", q, flags=re.I).strip()
    return q


def is_chitchat(q: str) -> bool:
    t = normalize_text(q)
    if not t:
        return True
    if t.lower() in CHITCHAT_EXACT or t in CHITCHAT_EXACT:
        return True
    if GRAPH_HINT.search(t):
        return False
    return len(t) < 12


def extract_cypher(raw: str) -> str:
    text = raw.strip()
    m = re.search(r"```(?:cypher)?\s*([\s\S]*?)```", text, re.I)
    if m:
        text = m.group(1).strip()
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    buf: list[str] = []
    started = False
    for line in lines:
        if not started and re.match(r"^(MATCH|CALL|WITH|UNWIND|RETURN)\b", line, re.I):
            started = True
        if started:
            buf.append(line)
    return ("\n".join(buf) if buf else text).rstrip(";").strip()


def split_md_chunks(text: str, soft_max: int = 2800) -> list[str]:
    parts = re.split(r"(?m)^##\s+", text)
    out: list[str] = []
    for i, p in enumerate(parts):
        block = (f"## {p}" if i > 0 else p).strip()
        if not block:
            continue
        if len(block) <= soft_max:
            out.append(block)
        else:
            for j in range(0, len(block), soft_max):
                out.append(block[j : j + soft_max])
    return out


def load_md_docs(md_dir: str) -> list[tuple[str, str]]:
    if not os.path.isdir(md_dir):
        return []
    docs: list[tuple[str, str]] = []
    for fn in sorted(os.listdir(md_dir)):
        if fn.endswith(".md"):
            docs.append((fn[:-3], os.path.join(md_dir, fn)))
    return docs


def pick_doc_paths(query: str, docs: list[tuple[str, str]]) -> list[str]:
    q = query.lower()
    qc = re.sub(r"[\s\-_]+", "", q)
    out: list[str] = []
    for stem, path in docs:
        s = stem.lower()
        sc = re.sub(r"[\s\-_]+", "", s)
        if s in q or sc in qc:
            out.append(path)
    return out


def score_chunk(chunk: str, query: str) -> int:
    q_tokens = set(re.findall(r"[a-zA-Z]{3,}|[\u4e00-\u9fff]{2,}", query.lower()))
    c = chunk.lower()
    return sum(2 for t in q_tokens if t in c)


class GraphMuseEngine:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kw = {"trust_remote_code": True, "device_map": "auto" if self.device == "cuda" else None}
        try:
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, dtype=self.dtype, **load_kw)
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=self.dtype, **load_kw)
        self.model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        self.model.eval()
        self.docs = load_md_docs(MD_DIR)

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512, temp: float = 0.2) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        dev = next(self.model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        gen_kw = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temp > 0,
            "temperature": max(temp, 1e-5),
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kw)
        plen = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(out[0, plen:], skip_special_tokens=True).strip()

    def run_cypher(self, cypher: str) -> list[dict[str, Any]]:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                return [dict(r) for r in session.run(cypher)]

    def retrieve_doc_excerpts(self, query: str, max_chunks: int = 4, max_chars: int = 7000) -> list[dict[str, str]]:
        paths = pick_doc_paths(query, self.docs)
        scored: list[tuple[int, str, str]] = []
        for path in paths:
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                continue
            base = os.path.basename(path)
            for chunk in split_md_chunks(text):
                scored.append((score_chunk(chunk, query), base, chunk))
        scored.sort(key=lambda x: -x[0])
        used = 0
        out: list[dict[str, str]] = []
        for _, source, chunk in scored:
            if len(out) >= max_chunks or used >= max_chars:
                break
            room = max_chars - used
            piece = chunk[:room].strip()
            if not piece:
                continue
            out.append({"source": source, "text": piece})
            used += len(piece)
        return out

    def answer(self, user_message: str, debug: bool = False) -> ChatResponse:
        query = normalize_text(user_message)
        if is_chitchat(query):
            text = self.generate(CHITCHAT_SYSTEM_PROMPT, query, max_new_tokens=180, temp=0.4)
            return ChatResponse(answer=text, mode="chitchat")

        raw_cypher = self.generate(CYPHER_SYSTEM_PROMPT, query, max_new_tokens=360, temp=0.1)
        cypher = extract_cypher(raw_cypher)
        rows: list[dict[str, Any]] = []
        if cypher and not FORBIDDEN_CYPHER.search(cypher):
            try:
                rows = self.run_cypher(cypher)
            except Exception:
                rows = []

        excerpts = self.retrieve_doc_excerpts(query)
        payload = {
            "user_question": query,
            "row_count": len(rows),
            "rows_json": rows[:50],
            "retrieved_doc_excerpts": excerpts,
        }
        text = self.generate(
            ANSWER_SYSTEM_PROMPT,
            "请基于下面 JSON 回答用户问题：\n\n" + json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            max_new_tokens=900,
            temp=0.25,
        )
        return ChatResponse(
            answer=text,
            mode="graph_rag",
            cypher=cypher if debug else None,
            row_count=len(rows),
        )


app = FastAPI(title=APP_NAME, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINE: GraphMuseEngine | None = None


@app.on_event("startup")
def _startup() -> None:
    global ENGINE
    ENGINE = GraphMuseEngine()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "name": APP_NAME}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    assert ENGINE is not None
    return ENGINE.answer(req.message, debug=req.debug)


_frontend_dir = "/root/llm/gfm_paper/app/frontend"
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
