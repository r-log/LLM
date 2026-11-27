"""Evaluate saved runs and print them side by side.

    python scripts/evaluate.py --data data/fineweb-edu runs/baseline-*/
    python scripts/evaluate.py --summary runs/*/      # just read final.json, no recompute
    python scripts/evaluate.py --data data/fineweb-edu --stride 0.25 runs/*/   # sliding evaluation

--stride S scores with overlapping windows, so every byte (after the first window) gets at
least block_size - S positions of context. S below 1 is a fraction of each model's own
block size (0.25 = a quarter), which keeps the comparison fair between models with different
block sizes or tokenizers. Results go to final_stride<S>.json (final.json, written by
train.py, keeps the plain non-overlapping score). --summary --stride S reads those.

With several seeds of the same setup, also prints mean and spread, so you can see
whether a difference between two setups is bigger than the noise.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.evaluation import load_val_docs, run_suite  # noqa: E402
from llm.model import GPT, GPTConfig  # noqa: E402
from llm.tokenizers import get_tokenizer  # noqa: E402


def result_file(stride):
    return f"final_stride{stride:g}.json" if stride else "final.json"


def evaluate_run(run_dir: Path, data: str, device: str, stride: int | None) -> dict:
    ckpt = torch.load(run_dir / "model.pt", map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    block = model.cfg.block_size
    s = None
    if stride:
        s = max(1, int(block * stride)) if stride < 1 else min(int(stride), block)
    results = run_suite(model, get_tokenizer(ckpt["tokenizer"]), load_val_docs(data), device, stride=s)
    results["stride"] = s
    (run_dir / result_file(stride)).write_text(json.dumps(results, indent=2))
    print(f"evaluated {run_dir.name} (block {block}, stride {s})", file=sys.stderr)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--data", help="data folder; recompute the evaluation suite")
    ap.add_argument("--summary", action="store_true", help="only read existing result files")
    ap.add_argument("--stride", type=float, help="sliding evaluation stride: positions, or a fraction (<1) of the block size")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not args.summary and not args.data:
        ap.error("pass --data to recompute, or --summary to read existing results")

    rows = []
    for run in args.runs:
        f = run / result_file(args.stride)
        if args.summary:
            if not f.exists():
                print(f"skipping {run.name}: no {f.name}", file=sys.stderr)
                continue
            res = json.loads(f.read_text())
        else:
            if not (run / "model.pt").exists():
                print(f"skipping {run.name}: no model.pt", file=sys.stderr)
                continue
            res = evaluate_run(run, args.data, args.device, args.stride)
        rows.append((run.name, res))
    if not rows:
        sys.exit("no results")

    # Only columns every run has (older runs may lack newer test sets).
    keys = [k for k in rows[0][1] if isinstance(rows[0][1][k], dict) and all(k in r for _, r in rows)]
    print(f"{'run':40s}" + "".join(f"{k:>16s}" for k in keys))
    for name, res in rows:
        print(f"{name:40s}" + "".join(f"{res[k]['bpb']:16.4f}" for k in keys))

    # Group seeds: "gpt2-small-s1", "gpt2-small-s2" -> "gpt2-small"
    groups = defaultdict(list)
    for name, res in rows:
        groups[re.sub(r"-s\d+$", "", name)].append(res)
    multi = {g: r for g, r in groups.items() if len(r) > 1}
    if multi:
        print("\nmean ± std over seeds (a difference smaller than ~2 std is probably noise):")
        for g, rs in multi.items():
            cells = []
            for k in keys:
                v = [r[k]["bpb"] for r in rs]
                cells.append(f"{statistics.mean(v):.4f}±{statistics.stdev(v):.4f}")
            print(f"{g + f' (n={len(rs)})':40s}" + "".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    main()
