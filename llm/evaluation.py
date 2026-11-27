"""The evaluation suite: the same checks for every model.

    clean        bits per byte on the fixed validation documents
    typos_X      bits per byte on the same documents with typo rate X (see typos.py)

For the typo sets, look at how much each model *degrades* relative to its own
clean score, not at the raw number: corrupted text is harder to predict for
everyone. A model that reads text more robustly should degrade less.
"""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import bits_per_byte
from .typos import add_typos

TYPO_RATES = (0.03, 0.10)
TYPO_SEED = 1234  # fixed forever, so every model sees the same typos


def load_val_docs(data_dir: str | Path, limit: int | None = None) -> list[str]:
    with open(Path(data_dir) / "val.jsonl", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    return docs[:limit] if limit else docs


def typo_docs(docs: list[str], rate: float) -> list[str]:
    return [add_typos(d, rate, seed=TYPO_SEED + i) for i, d in enumerate(docs)]


def run_suite(model, tokenizer, docs: list[str], device="cpu") -> dict:
    results = {"clean": bits_per_byte(model, docs, tokenizer, device)}
    for rate in TYPO_RATES:
        r = bits_per_byte(model, typo_docs(docs, rate), tokenizer, device)
        r["degradation"] = r["bpb"] / results["clean"]["bpb"] - 1  # e.g. 0.12 = 12% worse
        results[f"typos_{rate}"] = r
    return results


def summarize(results: dict) -> str:
    parts = [f"clean {results['clean']['bpb']:.4f}"]
    for k, r in results.items():
        if k.startswith("typos_"):
            parts.append(f"{k} {r['bpb']:.4f} (+{100 * r['degradation']:.1f}%)")
    return " | ".join(parts)
