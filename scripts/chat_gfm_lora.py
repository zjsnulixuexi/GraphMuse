#!/usr/bin/env python3
"""
交互式推理：基座 Qwen2.5 + 训练得到的 LoRA 适配器（未与基座合并，运行时动态相加）。

保存的 `adapter_model.safetensors` 只是增量；本脚本每次启动会加载 base + adapter。
若需要「单文件合并权重」，见文末说明或加 --merge_and_save（可选）。
"""

from __future__ import annotations

import os

os.environ["MKL_THREADING_LAYER"] = "GNU"

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat loop: Qwen2.5 base + GFM LoRA adapter.")
    p.add_argument("--base_model", type=str, default="/root/llm/model/qwen")
    p.add_argument("--adapter_path", type=str, default="/root/llm/gfm_paper/outputs/qwen25_gfm_lora")
    p.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are a graph foundation model (GFM) expert. "
            "You translate natural language questions about graph-structured knowledge into precise Neo4j Cypher."
        ),
    )
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--stream", action="store_true", help="Stream tokens to stdout as they generate.")
    p.add_argument(
        "--merge_and_save",
        type=str,
        default="",
        metavar="DIR",
        help="If set: merge LoRA into base and save full weights to DIR, then exit (no chat).",
    )
    return p.parse_args()


def build_prompt(tokenizer, system_prompt: str, user_text: str) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    if args.merge_and_save:
        merged = model.merge_and_unload()
        os.makedirs(args.merge_and_save, exist_ok=True)
        merged.save_pretrained(args.merge_and_save)
        tokenizer.save_pretrained(args.merge_and_save)
        print(f"Merged full model saved to: {args.merge_and_save}")
        return

    print("输入 quit / exit / q 结束。\n")

    while True:
        try:
            user = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit", "q"):
            break

        prompt = build_prompt(tokenizer, args.system_prompt, user)
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_kw = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        if args.stream:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kw["streamer"] = streamer
            thread = Thread(
                target=model.generate,
                kwargs={**inputs, **gen_kw},
            )
            thread.start()
            print("Assistant> ", end="", flush=True)
            for text in streamer:
                print(text, end="", flush=True)
            print()
            thread.join()
        else:
            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kw)
            # 只解码新生成段（去掉 prompt）
            prompt_len = inputs["input_ids"].shape[1]
            text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
            print(f"Assistant> {text.strip()}\n")


if __name__ == "__main__":
    main()
