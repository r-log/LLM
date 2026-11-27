"""Evaluate saved runs and print them side by side.

    python scripts/evaluate.py --data data/fineweb-edu runs/baseline-*/
    python scripts/evaluate.py --summary runs/*/      # just read final.json, no recompute

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


def evaluate_run(run_dir: Path, data: str, device: str) -> dict:
    ckpt = torch.load(run_dir / "model.pt", map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    results = run_suite(model, get_tokenizer(ckpt["tokenizer"]), load_val_docs(data), device)
    (run_dir / "final.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--data", help="data folder; recompute the evaluation suite")
    ap.add_argument("--summary", action="store_true", help="only read existing final.json files")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not args.summary and not args.data:
        ap.error("pass --data to recompute, or --summary to read existing results")

    rows = []
    for run in args.runs:
        res = json.loads((run / "final.json").read_text()) if args.summary else evaluate_run(run, args.data, args.device)
        rows.append((run.name, res))

    keys = [k for k in rows[0][1] if isinstance(rows[0][1][k], dict)]
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
