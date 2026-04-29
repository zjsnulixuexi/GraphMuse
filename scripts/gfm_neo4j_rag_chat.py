#!/usr/bin/env python3
"""
GraphRAG + 本地 MD 摘录：微调 Qwen+LoRA 生成 Cypher → Neo4j → 从 gfm_paper/md/*.md 检索相关段落
→ 本地 Qwen 综合「图查询结果 + 文档摘录」生成回答。

是否必须用 LangChain：不必。MD 侧使用轻量分块 + 关键词打分（无需向量库）；需要更强检索时可自行换 embedding。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch
from neo4j import GraphDatabase
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# 阶段 2：用本地 Qwen 把「问题 + Cypher + 查询结果」整理成给用户看的规范回答
# （与阶段 1 的 Cypher 生成任务不同，建议用偏「说明/归纳」的系统提示）
# ---------------------------------------------------------------------------
DEFAULT_ANSWER_SYSTEM_PROMPT = """\
你是「图数据库 + 论文笔记」综合问答助手。你会收到 JSON，其中：
- rows_json：Neo4j 结构化查询结果（可能为空）；
- retrieved_doc_excerpts：从项目内 Markdown 笔记中截取的文献/模型说明片段（可能为空）；
- neo4j_error：若图查询失败会有错误说明，否则为 null。

硬性规则：
1. 结构化事实（作者列表、数据集名、关系计数等）以 rows_json 为准；不得编造图中不存在的实体或数值。
2. 若用户问「论文/模型讲了什么、方法、贡献、背景」等叙述性问题，应优先结合 retrieved_doc_excerpts 作答；图结果仅作补充（如列出作者、数据集）。
3. 若 rows_json 为空且摘录也不足，应如实说明信息不足，不要臆造实验细节。
4. 若结果有多条，先概括再列要点；避免整表照搬。
5. 禁止在回答中出现任何 Cypher、MATCH、关系箭头 `->`、代码块或反引号。
6. 使用简体中文，语气专业、简洁。"""


# 闲聊 / 非图库问题：不调用 Neo4j，由模型简短自然语言回复
DEFAULT_CHITCHAT_SYSTEM_PROMPT = """\
你是友好的学术助手。用户的问题与「论文/模型/图数据库」无关（例如打招呼、寒暄、无关闲聊）。
请用一两句简体中文自然回复；不要生成 Cypher，不要提及 Neo4j，不要假装查过数据库。"""


DEFAULT_CYPHER_SYSTEM_PROMPT = (
    "You are a graph foundation model (GFM) expert. "
    "You translate natural language questions about graph-structured knowledge into precise Neo4j Cypher. "
    "Output only the Cypher query, no markdown code fences, no explanation."
)


# 命中任一即视为「可能与图/论文语料相关」，才进入 Neo4j 流程
_GRAPH_HINT = re.compile(
    r"(论文|刊物|题目|标题|作者|署名|模型|数据集|数据\s*集|指标|评测|概念|关系|引用|对比|"
    r"讲了什么|主要内容|概述|总结|贡献|方法|背景|提出|工作|"
    r"Cypher|Neo4j|图数据库|三元组|节点|边|Graph|GFM|知识图谱|Dataset|Metric|Concept|Paper|Author|Model|"
    r"MDGPT|RAG|GraphGPT|Cora|CiteSeer|PubMed|ogbn|arxiv|MultiGPrompt|GraphMORE|LLaGa|SAMGPT|GraphPrompt|OFA)",
    re.I,
)

_CHITCHAT_EXACT = {
    "你好",
    "您好",
    "哈喽",
    "在吗",
    "在么",
    "嗨",
    "hi",
    "hello",
    "hey",
    "谢谢",
    "多谢",
    "感谢",
    "thanks",
    "thank you",
    "再见",
    "拜拜",
    "bye",
    "早",
    "晚安",
    "?",
    "？",
    "你是谁",
    "你是",
    "帮助",
    "help",
    "怎么用",
}


def normalize_user_input(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"^User>\s*", "", q, flags=re.I).strip()
    return q


def should_skip_neo4j(user_q: str) -> bool:
    """闲聊、过短且无图语义的输入不查 Neo4j。"""
    t = normalize_user_input(user_q)
    if not t:
        return True
    low = t.lower()
    if t in _CHITCHAT_EXACT or low in _CHITCHAT_EXACT:
        return True
    if _GRAPH_HINT.search(t):
        return False
    # 含典型模型/数据集英文 token 时仍查库
    if re.search(r"\b(MDGPT|RAG[\s-]?GFM|GraphGPT|Neo4j|cypher)\b", t, re.I):
        return False
    if len(t) < 16:
        return True
    return False


def strip_leading_cypher_in_answer(text: str) -> str:
    """若模型仍输出以 MATCH 开头的残留，去掉其前段直至首个汉字（保守清理）。"""
    t = text.strip()
    if not re.match(r"^(MATCH|CALL|WITH|UNWIND)\b", t, re.I):
        return t
    m = re.search(r"[\u4e00-\u9fff]", t)
    if m and m.start() > 0:
        return t[m.start() :].lstrip(" ，。、\n")
    return t


_SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MD_DIR = os.path.abspath(os.path.join(_SCRIPT_ROOT, "..", "md"))


def discover_md_docs(md_dir: str) -> list[tuple[str, str]]:
    """扫描 md 目录，返回 (文件名 stem, 绝对路径)，stem 长的优先（减少短名误匹配）。"""
    if not os.path.isdir(md_dir):
        return []
    pairs: list[tuple[str, str]] = []
    for fn in sorted(os.listdir(md_dir)):
        if not fn.endswith(".md"):
            continue
        stem = fn[:-3]
        pairs.append((stem, os.path.join(md_dir, fn)))
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


def pick_doc_paths(user_q: str, docs: list[tuple[str, str]]) -> list[str]:
    """根据问题中出现的模型/笔记名，选择要检索的 md 文件路径（去重保持顺序）。"""
    q_low = user_q.lower()
    q_compact = re.sub(r"[\s\-_]+", "", q_low)
    out: list[str] = []
    seen: set[str] = set()
    for stem, path in docs:
        s_low = stem.lower()
        s_compact = re.sub(r"[\s\-_]+", "", s_low)
        if len(s_compact) < 2:
            continue
        if s_low in q_low or s_compact in q_compact:
            if path not in seen:
                out.append(path)
                seen.add(path)
    return out


def split_md_chunks(text: str, soft_max: int = 3200) -> list[str]:
    """按二级标题或长段切块，便于打分截断。"""
    parts = re.split(r"(?m)^##\s+", text)
    chunks: list[str] = []
    for i, p in enumerate(parts):
        block = f"## {p}" if i > 0 else p
        block = block.strip()
        if not block:
            continue
        if len(block) <= soft_max:
            chunks.append(block)
            continue
        for j in range(0, len(block), soft_max):
            chunks.append(block[j : j + soft_max])
    return chunks if chunks else [text[:soft_max]]


def _score_chunk(chunk: str, query: str, row_blob: str) -> int:
    bag = (query + " " + row_blob).lower()
    score = 0
    for tok in set(re.findall(r"[a-zA-Z]{3,}|[\u4e00-\u9fff]{2,}", bag)):
        if tok in chunk.lower():
            score += 2
    score += min(len(chunk) // 6000, 3)
    return score


def retrieve_md_excerpts(
    paths: list[str],
    user_q: str,
    rows: list[dict[str, Any]],
    max_total_chars: int,
    max_chunks: int = 10,
) -> list[dict[str, str]]:
    """从命中的 md 中选取与问题/图结果最相关的若干片段。"""
    row_blob = " ".join(str(v) for row in rows[:15] for v in row.values())
    scored: list[tuple[int, str, str]] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        base = os.path.basename(path)
        for ch in split_md_chunks(content):
            scored.append((_score_chunk(ch, user_q, row_blob), base, ch))
    scored.sort(key=lambda x: -x[0])
    out: list[dict[str, str]] = []
    used = 0
    for _, base, ch in scored[: max_chunks * 3]:
        if used >= max_total_chars:
            break
        room = max_total_chars - used
        piece = ch[:room].strip()
        if not piece:
            continue
        out.append({"source": base, "text": piece})
        used += len(piece)
        if len(out) >= max_chunks:
            break
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GFM: NL → Cypher (LoRA) → Neo4j → Qwen answer.")
    p.add_argument("--base_model", type=str, default="/root/llm/model/qwen")
    p.add_argument("--adapter_path", type=str, default="/root/llm/gfm_paper/outputs/qwen25_gfm_lora")
    p.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687")
    p.add_argument("--neo4j_user", type=str, default="neo4j")
    p.add_argument("--neo4j_password", type=str, default="123456zzz")
    p.add_argument("--cypher_system_prompt", type=str, default=DEFAULT_CYPHER_SYSTEM_PROMPT)
    p.add_argument("--answer_system_prompt", type=str, default=DEFAULT_ANSWER_SYSTEM_PROMPT)
    p.add_argument("--chitchat_system_prompt", type=str, default=DEFAULT_CHITCHAT_SYSTEM_PROMPT)
    p.add_argument("--max_new_tokens_cypher", type=int, default=512)
    p.add_argument("--max_new_tokens_answer", type=int, default=1024)
    p.add_argument("--temperature_cypher", type=float, default=0.1)
    p.add_argument("--temperature_answer", type=float, default=0.3)
    p.add_argument("--max_result_rows", type=int, default=80, help="写入阶段2提示的最大行数，防止上下文过长")
    p.add_argument("--debug", action="store_true", help="仅调试时打印阶段1 Cypher（正常对话不展示）")
    p.add_argument(
        "--no_skip_neo4j",
        action="store_true",
        help="关闭闲聊检测，所有输入都走 Neo4j（调试用）",
    )
    p.add_argument(
        "--md_dir",
        type=str,
        default=_DEFAULT_MD_DIR,
        help="模型说明 Markdown 所在目录（默认 gfm_paper/md）",
    )
    p.add_argument(
        "--md_max_chars",
        type=int,
        default=20000,
        help="写入阶段2的 MD 摘录总字符上限",
    )
    p.add_argument(
        "--disable_md_rag",
        action="store_true",
        help="关闭 MD 检索，仅使用 Neo4j 结果",
    )
    return p.parse_args()


def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_reply(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    gen_kw = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kw)
    plen = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0, plen:], skip_special_tokens=True).strip()


def extract_cypher(raw: str) -> str:
    """去掉 markdown 围栏与多余说明，尽量得到单条可执行 Cypher。"""
    t = raw.strip()
    fence = re.search(r"```(?:cypher)?\s*([\s\S]*?)```", t, re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    # 若仍含多段，取第一条以 MATCH / CALL / WITH 开头的语句块（简单启发式）
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    buf: list[str] = []
    started = False
    for ln in lines:
        if not started and re.match(r"^(MATCH|CALL|WITH|UNWIND|RETURN)\b", ln, re.I):
            started = True
        if started:
            buf.append(ln)
    if buf:
        return "\n".join(buf).strip().rstrip(";")
    return t.rstrip(";").strip()


_FORBIDDEN_CYPHER = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH\s+DELETE|SET|REMOVE|DROP\s+DATABASE|DROP\s+CONSTRAINT|"
    r"DROP\s+INDEX|LOAD\s+CSV|FOREACH|APOC\.(create|merge)|GRANT|DENY|REVOKE)\b",
    re.IGNORECASE | re.DOTALL,
)


def assert_read_only_cypher(cypher: str) -> None:
    if _FORBIDDEN_CYPHER.search(cypher):
        raise ValueError("出于安全考虑，已拒绝可能改写图结构的语句；请仅使用只读查询（MATCH/RETURN 等）。")


def run_cypher(uri: str, user: str, password: str, cypher: str) -> list[dict[str, Any]]:
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            result = session.run(cypher)
            return [dict(r) for r in result]


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw = dict(
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    try:
        base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype, **load_kw)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, **load_kw)
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    md_docs = discover_md_docs(args.md_dir)
    if md_docs and not args.disable_md_rag:
        print(
            f"GraphRAG+MD 就绪：已发现 {len(md_docs)} 个 md 笔记；"
            "闲聊不查库；最终回答不展示 Cypher（--debug 可看）。输入 quit 退出。\n"
        )
    else:
        print("GraphRAG 就绪（未启用 MD 目录或已关闭 RAG）。输入 quit 退出。\n")

    while True:
        try:
            user_q = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_q:
            continue
        if user_q.lower() in ("quit", "exit", "q"):
            break

        user_q = normalize_user_input(user_q)

        if not args.no_skip_neo4j and should_skip_neo4j(user_q):
            ch_prompt = build_chat_prompt(tokenizer, args.chitchat_system_prompt, user_q)
            answer = generate_reply(
                model,
                tokenizer,
                ch_prompt,
                max_new_tokens=256,
                temperature=args.temperature_answer,
            )
            print(f"Assistant> {strip_leading_cypher_in_answer(answer)}\n")
            continue

        # --- 阶段 1：生成 Cypher（不向用户打印）---
        cypher_prompt = build_chat_prompt(tokenizer, args.cypher_system_prompt, user_q)
        raw_cypher = generate_reply(
            model,
            tokenizer,
            cypher_prompt,
            max_new_tokens=args.max_new_tokens_cypher,
            temperature=args.temperature_cypher,
        )
        cypher = extract_cypher(raw_cypher)
        if args.debug:
            print(f"[debug] raw:\n{raw_cypher}\n[debug] extracted:\n{cypher}\n")

        rows: list[dict[str, Any]] = []
        neo4j_error: str | None = None
        try:
            if not cypher.strip():
                raise ValueError("模型未生成有效 Cypher")
            assert_read_only_cypher(cypher)
            rows = run_cypher(args.neo4j_uri, args.neo4j_user, args.neo4j_password, cypher)
        except Exception as e:
            neo4j_error = str(e)
            if args.debug:
                print(f"[debug] Neo4j 执行失败（仍将尝试结合 MD 作答）：{neo4j_error}\n")

        doc_paths = pick_doc_paths(user_q, md_docs) if not args.disable_md_rag else []
        excerpts: list[dict[str, str]] = []
        if doc_paths:
            excerpts = retrieve_md_excerpts(doc_paths, user_q, rows, args.md_max_chars)
        elif args.debug and not args.disable_md_rag:
            print(f"[debug] 未匹配到 md 文件（md_dir={args.md_dir}）。\n")

        trimmed = rows[: args.max_result_rows]
        payload: dict[str, Any] = {
            "user_question": user_q,
            "row_count": len(rows),
            "rows_json": trimmed,
            "retrieved_doc_excerpts": excerpts,
            "neo4j_error": neo4j_error,
        }
        user_stage2 = (
            "请根据下面 JSON 回答用户问题。字段说明："
            "user_question=用户原问；row_count=图库命中行数；rows_json=图查询行；"
            "retrieved_doc_excerpts=从对应 md 笔记截取的片段；neo4j_error=图查询失败原因（成功则为 null）。\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
        )
        answer_prompt = build_chat_prompt(tokenizer, args.answer_system_prompt, user_stage2)
        answer = generate_reply(
            model,
            tokenizer,
            answer_prompt,
            max_new_tokens=args.max_new_tokens_answer,
            temperature=args.temperature_answer,
        )
        print(f"Assistant> {strip_leading_cypher_in_answer(answer)}\n")


if __name__ == "__main__":
    main()
