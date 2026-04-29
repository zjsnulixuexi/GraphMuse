#!/usr/bin/env python3
"""
GFM SFT: Qwen2.5-7B-Instruct + native PEFT LoRA (transformers + peft only).

Hardware: Tesla V100S 32GB — fp16 only, no bfloat16.
Chat formatting: tokenizer.apply_chat_template (Qwen2.5 ChatML, <|im_start|> / template eos).
"""

from __future__ import annotations

import os

# MKL / OpenMP threading conflict mitigation (must run before numpy / torch import).
os.environ["MKL_THREADING_LAYER"] = "GNU"

import argparse
import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, Features, Sequence, Value, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

# Force int64 token columns so Arrow never stores token lists as strings (fixes tokenizer.pad errors).
SFT_TOKEN_FEATURES = Features(
    {
        "input_ids": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
    }
)


def _coerce_int64_token_list(x: Any) -> list[int]:
    """Normalize dataset / Arrow quirks (e.g. nested ids stored as str) to list[int]."""
    if x is None:
        return []
    if isinstance(x, str):
        x = json.loads(x)
    if hasattr(x, "tolist"):
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        raise TypeError(f"Expected token id sequence, got {type(x)}")
    return [int(v) for v in x]


@dataclass
class DataCollatorForCausalLMSFT:
    """Pad variable-length SFT batches; keep causal LM labels (-100 on prompt + pad)."""

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [_coerce_int64_token_list(f["input_ids"]) for f in features]
        labels = [_coerce_int64_token_list(f["labels"]) for f in features]

        input_padded = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # Labels contain -100; pad only with -100 (do not run through tokenizer.pad).
        lab = torch.full(input_padded["input_ids"].shape, -100, dtype=torch.long)
        for i, y in enumerate(labels):
            L = min(len(y), lab.shape[1])
            lab[i, :L] = torch.tensor(y[:L], dtype=torch.long)
        return {"input_ids": input_padded["input_ids"], "attention_mask": input_padded["attention_mask"], "labels": lab}


def discover_linear_target_modules(model: torch.nn.Module, exclude_lm_head: bool) -> list[str]:
    """Collect last name segments of all nn.Linear submodules for LoRA target_modules."""
    seen: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if exclude_lm_head and name.endswith("lm_head"):
            continue
        seen.add(name.split(".")[-1])
    return sorted(seen)


def build_user_content(example: dict) -> str:
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    if inp:
        return f"{instruction}\n\n{inp}"
    return instruction


_apply_chat_template_has_return_dict: bool | None = None


def _chat_template_token_ids(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
    """Transformers 5.2+ returns BatchEncoding when tokenize=True; never use list(batch) (that yields key strings)."""
    global _apply_chat_template_has_return_dict
    if _apply_chat_template_has_return_dict is None:
        _apply_chat_template_has_return_dict = (
            "return_dict" in inspect.signature(tokenizer.apply_chat_template).parameters
        )
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_tensors": None,
    }
    if _apply_chat_template_has_return_dict:
        kwargs["return_dict"] = False
    out = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(out, list) and (not out or isinstance(out[0], int)):
        return out
    if isinstance(out, dict):
        return [int(x) for x in out["input_ids"]]
    return [int(x) for x in out["input_ids"]]


def tokenize_chat_example(
    example: dict,
    tokenizer,
    max_length: int,
    system_prompt: str | None,
):
    user_text = build_user_content(example)
    output = (example.get("output") or "").strip()

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": output})

    prompt_messages = messages[:-1]
    full_messages = messages

    prompt_ids = _chat_template_token_ids(tokenizer, prompt_messages, add_generation_prompt=True)

    full_ids = _chat_template_token_ids(tokenizer, full_messages, add_generation_prompt=False)

    if len(full_ids) < len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        # Rare tokenizer edge cases: fall back to mask by re-tokenizing assistant span only.
        logger.warning("Prompt ids not a prefix of full ids; masking assistant-only span heuristically.")
        assistant_only = _chat_template_token_ids(
            tokenizer,
            [{"role": "assistant", "content": output}],
            add_generation_prompt=False,
        )
        # Strip leading special duplication if any — keep last len(assistant_only) of full as trainable
        cut = max(0, len(full_ids) - len(assistant_only))
        prompt_ids = full_ids[:cut]

    labels = list(full_ids)
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100

    # Truncate from the left if needed (keep tail for answer + recent context).
    if len(full_ids) > max_length:
        drop = len(full_ids) - max_length
        full_ids = full_ids[drop:]
        labels = labels[drop:]

    # Plain Python int so Arrow / datasets map never infers string columns.
    full_ids = [int(t) for t in full_ids]
    labels = [int(t) for t in labels]

    return {"input_ids": full_ids, "labels": labels}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA SFT for Qwen2.5 GFM on JSON instruction data.")
    p.add_argument("--model_name_or_path", type=str, default="/root/llm/model/qwen")
    p.add_argument("--train_file", type=str, default="/root/llm/gfm_paper/data/gfm_sft.json")
    p.add_argument("--output_dir", type=str, default="/root/llm/gfm_paper/outputs/qwen25_gfm_lora")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--eval_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--exclude_lm_head_from_lora",
        action="store_true",
        help="If set, do not attach LoRA to lm_head (saves VRAM; still covers all transformer Linear).",
    )

    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--early_stopping_patience", type=int, default=3)

    p.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are a graph foundation model (GFM) expert. "
            "You translate natural language questions about graph-structured knowledge into precise Neo4j Cypher."
        ),
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    set_seed(args.seed)

    raw = load_dataset("json", data_files=args.train_file, split="train")
    split = raw.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_ds: Dataset = split["train"]
    eval_ds: Dataset = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def _tok(batch):
        # HuggingFace map passes batched columns; we expand per row for clarity.
        keys = batch.keys()
        n = len(batch[next(iter(keys))])
        input_ids_list, labels_list = [], []
        for i in range(n):
            ex = {k: batch[k][i] for k in keys}
            out = tokenize_chat_example(
                ex, tokenizer, max_length=args.max_length, system_prompt=args.system_prompt
            )
            input_ids_list.append(out["input_ids"])
            labels_list.append(out["labels"])
        return {"input_ids": input_ids_list, "labels": labels_list}

    map_kw = {
        "batched": True,
        "batch_size": 16,
        "remove_columns": train_ds.column_names,
        "features": SFT_TOKEN_FEATURES,
    }
    train_tok = train_ds.map(_tok, **map_kw)
    map_kw_eval = {**map_kw, "remove_columns": eval_ds.column_names}
    eval_tok = eval_ds.map(_tok, **map_kw_eval)

    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    target_modules = discover_linear_target_modules(model, exclude_lm_head=args.exclude_lm_head_from_lora)
    logger.info("LoRA target_modules (%d): %s", len(target_modules), target_modules)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForCausalLMSFT(tokenizer=tokenizer, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)

    # transformers>=5.2: tokenizer -> processing_class
    trainer_kw: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_tok,
        "eval_dataset": eval_tok,
        "data_collator": data_collator,
        "callbacks": [early_stopping],
    }
    _tsig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in _tsig:
        trainer_kw["processing_class"] = tokenizer
    elif "tokenizer" in _tsig:
        trainer_kw["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kw)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "model_name_or_path": args.model_name_or_path,
        "train_file": args.train_file,
        "lora_target_modules": target_modules,
        "exclude_lm_head_from_lora": args.exclude_lm_head_from_lora,
        "max_length": args.max_length,
        "training_args": training_args.to_dict(),
    }
    with open(os.path.join(args.output_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    if trainer.state.log_history:
        with open(os.path.join(args.output_dir, "log_history.json"), "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, indent=2, default=str)


if __name__ == "__main__":
    main()
