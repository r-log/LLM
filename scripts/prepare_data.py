"""Download text and turn it into training files.

Writes, under data/<source>/:
    val.jsonl          the fixed validation documents, as raw text (one JSON string per line).
                       Raw text, not tokens, so any tokenizer (or byte model) is scored on
                       exactly the same bytes.
    train_<tok>.bin    the training documents as token ids (uint16), one file per tokenizer,
                       each document preceded by that tokenizer's end-of-text id.
    meta.json          counts: documents, bytes, tokens per tokenizer.

The validation set is always the *first* N documents of the source, so it stays the same
no matter how much training data you prepare later.

Sources:
    fineweb-edu   HuggingFace FineWeb-Edu "sample-10BT" (needs internet; use on the GPU box)
    stdlib        English docstrings from Python's own standard library (works offline;
                  only for smoke tests, it is far too small and odd for real results)

Examples:
    python scripts/prepare_data.py --source stdlib
    python scripts/prepare_data.py --source fineweb-edu --train-bytes 2e9
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import sysconfig
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.tokenizers import get_tokenizer  # noqa: E402


def stdlib_docs():
    """Yield docstrings from the standard library, read with `ast` (nothing is imported)."""
    root = Path(sysconfig.get_paths()["stdlib"])
    for path in sorted(root.rglob("*.py")):
        if any(p in {"test", "tests", "idlelib", "site-packages"} for p in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if doc and len(doc) >= 200:
                    yield doc


def fineweb_docs():
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for row in ds:
        yield row["text"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["fineweb-edu", "stdlib"], required=True)
    ap.add_argument("--tokenizers", default="bytes,gpt2", help="comma-separated: bytes,gpt2")
    ap.add_argument("--val-docs", type=int, default=None, help="default: 2000 (fineweb-edu), 300 (stdlib)")
    ap.add_argument("--train-bytes", type=float, default=float("inf"), help="stop after this many training bytes")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    n_val = args.val_docs or (2000 if args.source == "fineweb-edu" else 300)
    out = Path(args.out) / args.source
    out.mkdir(parents=True, exist_ok=True)
    toks = []
    for name in args.tokenizers.split(","):
        try:
            toks.append(get_tokenizer(name))
        except Exception as e:  # e.g. gpt2 needs internet the first time
            print(f"skipping tokenizer {name!r}: {type(e).__name__}: {e}".splitlines()[0])
    if not toks:
        sys.exit("no usable tokenizer")

    docs = stdlib_docs() if args.source == "stdlib" else fineweb_docs()
    files = {t.name: open(out / f"train_{t.name}.bin", "wb") for t in toks}
    counts = {t.name: 0 for t in toks}
    val, n_train_docs, n_train_bytes = [], 0, 0
    buffers = {t.name: [] for t in toks}

    def flush():
        for t in toks:
            if buffers[t.name]:
                np.concatenate(buffers[t.name]).tofile(files[t.name])
                buffers[t.name].clear()

    for doc in docs:
        if len(val) < n_val:
            val.append(doc)
            continue
        if n_train_bytes >= args.train_bytes:
            break
        for t in toks:
            if t.name == "bytes":  # fast path: the UTF-8 bytes already are the ids
                body = np.frombuffer(doc.encode("utf-8"), dtype=np.uint8).astype(np.uint16)
            else:
                body = np.asarray(t.encode(doc), dtype=np.uint16)
            buffers[t.name] += [np.array([t.eot_id], dtype=np.uint16), body]
            counts[t.name] += 1 + len(body)
        n_train_docs += 1
        n_train_bytes += len(doc.encode("utf-8"))
        if n_train_docs % 10_000 == 0:
            flush()
            print(f"{n_train_docs:,} docs, {n_train_bytes / 1e9:.3f} GB", flush=True)
    flush()
    for f in files.values():
        f.close()

    with open(out / "val.jsonl", "w", encoding="utf-8") as f:
        for doc in val:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    meta = {
        "source": args.source,
        "val_docs": len(val),
        "val_bytes": sum(len(d.encode("utf-8")) for d in val),
        "train_docs": n_train_docs,
        "train_bytes": n_train_bytes,
        "train_tokens": counts,
        "bytes_per_token": {k: n_train_bytes / max(v - n_train_docs, 1) for k, v in counts.items()},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    if n_train_docs == 0:
        sys.exit("no training documents left after the validation split")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
