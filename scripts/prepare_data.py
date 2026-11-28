"""Download text and turn it into training files.

Writes, under data/<source>/:
    val.jsonl          the fixed validation documents, as raw text (one JSON string per line).
                       Raw text, not tokens, so any tokenizer (or byte model) is scored on
                       exactly the same bytes.
    train_<tok>.bin    the training documents as token ids (uint16), one file per tokenizer,
                       each document preceded by that tokenizer's end-of-text id.
    meta.json          counts: documents, bytes, tokens per tokenizer.

The validation set is always the *first* N documents of the source (or of its validation
split), so it stays the same no matter how much training data you prepare later.

Sources:
    fineweb-edu   HuggingFace FineWeb-Edu "sample-10BT" (needs internet; use on the GPU box)
    code          repositories from scripts/build_code_corpus.py (data/code-raw/*.jsonl). Each
                  repository becomes one document: its files in dependency order, each
                  preceded by a 0x1C byte (ASCII file separator) and its path. Validation
                  repositories are cut to their first --val-doc-bytes bytes.
    pg19          PG-19: Project Gutenberg books from before 1919, ~400 KB each (long-range
                  text). Validation comes from the official validation split.
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


def pg19_docs(split: str):
    from datasets import load_dataset

    ds = load_dataset("emozilla/pg19", split=split, streaming=True)
    for row in ds:
        if "text" not in row:
            raise KeyError(f"pg19 rows have no 'text' field; fields are {list(row)}")
        yield row["text"]


FILE_SEP = "\x1c"


def render_repo(rec: dict) -> str:
    """One repository as one text: '<0x1C>path\n' + content for every file, in stored order."""
    parts = []
    for f in rec["files"]:
        text = f["text"] if f["text"].endswith("\n") else f["text"] + "\n"
        parts.append(f"{FILE_SEP}{f['path']}\n{text}")
    return "".join(parts)


def code_docs(split: str, raw_dir: str = "data/code-raw", max_bytes: int | None = None):
    with open(Path(raw_dir) / f"{split}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            doc = render_repo(json.loads(line))
            if max_bytes:
                doc = doc.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
            yield doc


# source -> (default number of validation docs, validation iterator or None, train iterator)
# With None, validation docs are taken from the start of the train iterator.
SOURCES = {
    "fineweb-edu": (2000, None, fineweb_docs),
    "stdlib": (300, None, stdlib_docs),
    "pg19": (10, lambda: pg19_docs("validation"), lambda: pg19_docs("train")),
    "code": (20, None, None),  # filled in main() (needs --code-raw and --val-doc-bytes)
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=list(SOURCES), required=True)
    ap.add_argument("--tokenizers", default="bytes,gpt2", help="comma-separated: bytes,gpt2")
    ap.add_argument("--val-docs", type=int, default=None, help="default: 2000 (fineweb-edu), 300 (stdlib), 10 books (pg19)")
    ap.add_argument("--train-bytes", type=float, default=float("inf"), help="stop after this many training bytes")
    ap.add_argument("--out", default="data")
    ap.add_argument("--code-raw", default="data/code-raw", help="input folder for --source code")
    ap.add_argument("--val-doc-bytes", type=int, default=1_000_000, help="code: cut validation repos to this size")
    args = ap.parse_args()

    default_val, val_iter, train_iter = SOURCES[args.source]
    if args.source == "code":
        val_iter = lambda: code_docs("val", args.code_raw, args.val_doc_bytes)  # noqa: E731
        train_iter = lambda: code_docs("train", args.code_raw)  # noqa: E731
    n_val = args.val_docs or default_val
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

    docs = train_iter()
    files = {t.name: open(out / f"train_{t.name}.bin", "wb") for t in toks}
    counts = {t.name: 0 for t in toks}
    val, n_train_docs, n_train_bytes = [], 0, 0
    if val_iter is not None:
        for doc in val_iter():
            if len(val) >= n_val:
                break
            val.append(doc)
    buffers = {t.name: [] for t in toks}
    buffered, next_report = 0, 0.1e9

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
        n_bytes = len(doc.encode("utf-8"))
        n_train_bytes += n_bytes
        buffered += n_bytes
        if buffered >= 50e6:  # flush by size: one book is ~0.4 MB, one web page ~5 KB
            flush()
            buffered = 0
        if n_train_bytes >= next_report:
            print(f"{n_train_docs:,} docs, {n_train_bytes / 1e9:.3f} GB", flush=True)
            next_report += 0.1e9
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
