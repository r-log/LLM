"""Per-byte surprise statistics for a trained byte-level model.

Surprise = the model's loss on a byte, in bits (how many yes/no questions it still needed).

    python scripts/surprise_stats.py runs/bytes-small-s1 --data data/fineweb-edu
    python scripts/surprise_stats.py runs/bytes-small-s1 --data data/fineweb-edu --show 3

Reports:
  concentration   what share of all bits comes from the most surprising X% of bytes
  histogram       how many bytes fall into each surprise range
  by byte kind    space, 1st/2nd/3rd/4th+ letter of a word, digit, punctuation, ...
  by context      surprise vs. how much context the byte had inside its window
  easy runs       how often long stretches of nearly-free bytes occur
  --show N        N text samples colored by surprise (green = easy, red = surprising)

Writes the numbers to <run>/surprise.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.evaluation import load_val_docs  # noqa: E402
from llm.metrics import per_token_nats  # noqa: E402
from llm.model import GPT, GPTConfig  # noqa: E402
from llm.tokenizers import get_tokenizer  # noqa: E402

EASY = 0.5  # bits; a byte below this is "nearly free"


def byte_kinds(text: str) -> list[str]:
    """One label per UTF-8 byte of `text`."""
    kinds, pos_in_word = [], 0
    for ch in text:
        if ch.isalpha():
            pos_in_word += 1
            k = f"letter {pos_in_word}" if pos_in_word <= 3 else "letter 4+"
        else:
            pos_in_word = 0
            if ch == " ":
                k = "space"
            elif ch == "\n":
                k = "newline"
            elif ch.isdigit():
                k = "digit"
            else:
                k = "punctuation/other"
        n = len(ch.encode("utf-8"))
        kinds += [k] + (["non-ASCII continuation"] * (n - 1) if n > 1 else [])
    return kinds


def colorize(text_bytes: bytes, bits: np.ndarray) -> str:
    out = []
    for b, v in zip(text_bytes, bits):
        ch = chr(b) if 32 <= b < 127 else ("⏎\n" if b == 10 else "·")
        color = 22 if v < EASY else 28 if v < 1 else 100 if v < 2 else 130 if v < 4 else 124
        out.append(f"\033[48;5;{color}m\033[97m{ch}\033[0m")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--data", required=True)
    ap.add_argument("--docs", type=int, default=None, help="limit the number of validation docs")
    ap.add_argument("--show", type=int, default=0, help="print N colored text samples")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.run / "model.pt", map_location=args.device)
    if ckpt["tokenizer"] != "bytes":
        sys.exit("surprise_stats works on byte-level models only (one score per byte)")
    model = GPT(GPTConfig(**ckpt["config"])).to(args.device)
    model.load_state_dict(ckpt["model"])
    tok = get_tokenizer("bytes")
    docs = load_val_docs(args.data, args.docs)

    nats, window_pos, n_bytes = per_token_nats(model, docs, tok, args.device)
    bits = (nats.double() / math.log(2)).numpy()
    window_pos = window_pos.numpy()
    kinds = np.array([k for d in docs for k in byte_kinds(d)])
    assert len(kinds) == len(bits) == n_bytes
    total = bits.sum()
    res = {"run": args.run.name, "bytes": int(n_bytes), "bpb": float(total / n_bytes)}
    print(f"{args.run.name}: {n_bytes:,} bytes, {res['bpb']:.4f} bits per byte\n")

    # 1. Concentration
    srt = np.sort(bits)[::-1]
    cum = np.cumsum(srt) / total
    res["share_of_bits_from_top"] = {}
    print("CONCENTRATION: share of all bits from the most surprising bytes")
    for pct in (1, 5, 10, 20, 50):
        share = float(cum[max(1, int(len(srt) * pct / 100)) - 1])
        res["share_of_bits_from_top"][f"{pct}%"] = share
        print(f"  top {pct:2d}% of bytes -> {100 * share:5.1f}% of bits")
    for target in (0.5, 0.8, 0.9):
        frac = float((np.searchsorted(cum, target) + 1) / len(srt))
        res[f"bytes_needed_for_{int(100 * target)}%_of_bits"] = frac
        print(f"  {100 * target:.0f}% of all bits come from {100 * frac:5.1f}% of bytes")

    # 2. Histogram
    print("\nHISTOGRAM: bytes by surprise")
    edges = [0, 0.1, 0.5, 1, 2, 4, 8, float("inf")]
    res["histogram"] = {}
    for lo, hi in zip(edges, edges[1:]):
        m = (bits >= lo) & (bits < hi)
        label = f"{lo:g}-{hi:g} bits" if hi != float("inf") else f">{lo:g} bits"
        res["histogram"][label] = {"bytes": float(m.mean()), "bits": float(bits[m].sum() / total)}
        print(f"  {label:12s} {100 * m.mean():5.1f}% of bytes, {100 * bits[m].sum() / total:5.1f}% of bits")

    # 3. By kind of byte
    print("\nBY BYTE KIND                    share of bytes   avg bits   share of bits")
    res["by_kind"] = {}
    order = ["space", "letter 1", "letter 2", "letter 3", "letter 4+", "digit", "punctuation/other",
             "newline", "non-ASCII continuation"]
    for k in order:
        m = kinds == k
        if not m.any():
            continue
        r = {"bytes": float(m.mean()), "avg_bits": float(bits[m].mean()), "bits": float(bits[m].sum() / total)}
        res["by_kind"][k] = r
        print(f"  {k:28s} {100 * r['bytes']:8.1f}%   {r['avg_bits']:8.2f}   {100 * r['bits']:8.1f}%")

    # 4. By context available
    print("\nBY CONTEXT (position inside the window)")
    res["by_context"] = {}
    T = model.cfg.block_size
    edges = [0, 8, 32, 128, 512, 2048, 8192, 32768]
    for lo, hi in zip(edges, edges[1:]):
        if lo >= T:
            break
        m = (window_pos >= lo) & (window_pos < hi)
        if m.any():
            res["by_context"][f"{lo}-{hi}"] = float(bits[m].mean())
            print(f"  {lo:5d}-{min(hi, T):<5d} bytes of context: {bits[m].mean():.3f} bits/byte")

    # 5. Runs of nearly-free bytes
    easy = bits < EASY
    runs, n = [], 0
    for e in easy:
        if e:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    runs = np.array(runs) if runs else np.array([0])
    res["easy_runs"] = {"threshold_bits": EASY, "mean_length": float(runs.mean()),
                        "bytes_in_runs_of_4+": float(runs[runs >= 4].sum() / n_bytes),
                        "bytes_in_runs_of_8+": float(runs[runs >= 8].sum() / n_bytes)}
    print(f"\nEASY RUNS (consecutive bytes under {EASY} bits)")
    print(f"  average run length {runs.mean():.2f} bytes")
    print(f"  {100 * res['easy_runs']['bytes_in_runs_of_4+']:.1f}% of all bytes are in runs of 4 or more")
    print(f"  {100 * res['easy_runs']['bytes_in_runs_of_8+']:.1f}% of all bytes are in runs of 8 or more")

    (args.run / "surprise.json").write_text(json.dumps(res, indent=2))

    # 6. Colored samples
    if args.show:
        print("\nSAMPLES  colors: green <0.5 bits, dark green <1, olive <2, orange <4, red >=4")
        offset = 0
        for i, d in enumerate(docs):
            b = d.encode("utf-8")
            if i < args.show:
                seg = slice(offset, offset + min(len(b), 600))
                print(f"\n--- doc {i} ---")
                print(colorize(b[:600], bits[seg]))
            offset += len(b)
    print(f"\nsaved {args.run / 'surprise.json'}")


if __name__ == "__main__":
    main()
