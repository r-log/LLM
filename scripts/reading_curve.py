"""Reading curve: bits per byte as a model reads further into each validation document
(for code: each repository), with or without learning while reading.

    # static: no learning (S = stride; default = block size, i.e. no overlap)
    python scripts/reading_curve.py runs/X --data data/code
    python scripts/reading_curve.py runs/X --data data/code --stride 0.25        # more context
    # learning while reading: after scoring each window, one optimizer step on ...
    python scripts/reading_curve.py runs/X --data data/code --select all --lr 1e-4
    python scripts/reading_curve.py runs/X --data data/code --select surprise --frac 0.2 --lr 1e-4
    python scripts/reading_curve.py runs/X --data data/code --select random --frac 0.2 --lr 1e-4
    # forgetting: bits per byte on other text before/after each adaptation
    ... --forget-data data/fineweb-edu --forget-docs 20

Results go to <run>/reading_<tag>.json. Compare runs on the same documents: per_doc_bpb
is stored, so differences can be paired by document.

Compute: a static pass costs about one forward pass per window. Adapting adds a backward
pass (~2x a forward), so roughly 3x per window (more with --steps > 1). The script prints
this factor so that adapting can be compared with spending the same compute on context.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.evaluation import load_val_docs  # noqa: E402
from llm.model import GPT, GPTConfig  # noqa: E402
from llm.reading import run  # noqa: E402
from llm.tokenizers import get_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--data", required=True, help="folder with val.jsonl (one document per line)")
    ap.add_argument("--docs", type=int, help="use only the first N documents")
    ap.add_argument("--stride", type=float, help="window step: positions, or a fraction (<1) of the block size")
    ap.add_argument("--select", choices=["all", "surprise", "random"], help="learn while reading (needs --lr)")
    ap.add_argument("--frac", type=float, default=0.2, help="share of tokens to learn from (surprise/random)")
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--params", default="all", help="all, blocks:N (last N blocks) or mlp")
    ap.add_argument("--steps", type=int, default=1, help="optimizer steps per window")
    ap.add_argument("--forget-data", help="folder with val.jsonl of other text, to measure forgetting")
    ap.add_argument("--forget-docs", type=int, default=20)
    ap.add_argument("--forget-bytes", type=int, default=65536, help="cut each forgetting document to this size")
    ap.add_argument("--tag", help="name for the output file (default: built from the settings)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.select and args.lr <= 0:
        ap.error("--select needs --lr > 0")

    ckpt = torch.load(args.run / "model.pt", map_location=args.device)
    model = GPT(GPTConfig(**ckpt["config"])).to(args.device)
    model.load_state_dict(ckpt["model"])
    tok = get_tokenizer(ckpt["tokenizer"])
    T = model.cfg.block_size
    stride = None
    if args.stride:
        stride = max(1, int(T * args.stride)) if args.stride < 1 else min(int(args.stride), T)
    docs = load_val_docs(args.data, args.docs)
    forget = None
    if args.forget_data:
        forget = [d.encode("utf-8")[: args.forget_bytes].decode("utf-8", errors="ignore")
                  for d in load_val_docs(args.forget_data, args.forget_docs)]

    res = run(model, docs, tok, forget_docs=forget, stride=stride, select=args.select, frac=args.frac,
              lr=args.lr, opt=args.opt, params=args.params, steps=args.steps, device=args.device)
    S = stride or T
    windows_per_token = T / S
    res["compute_per_token_vs_plain_forward"] = windows_per_token * (1 + (2 * args.steps + (args.steps - 1)) if args.select else 1)
    res["settings"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    tag = args.tag or (f"static-s{S}" if not args.select else
                       f"{args.select}{'' if args.select == 'all' else args.frac}-lr{args.lr:g}-{args.opt}-{args.params}-s{S}")
    tag = tag.replace(":", "")
    (args.run / f"reading_{tag}.json").write_text(json.dumps(res, indent=2))

    print(f"{args.run.name}  [{tag}]  {len(docs)} documents, {res['updates']} updates, "
          f"compute ~{res['compute_per_token_vs_plain_forward']:.1f}x a plain forward pass")
    print("bytes already read   bits/byte")
    for k, v in res["curve"].items():
        print(f"  {k:16s}  {v:8.4f}")
    if forget:
        f = res["forgetting"]
        print(f"forgetting: other text {f['before_bpb']:.4f} bpb before, +{f['mean_increase']:.4f} on average after adapting")
    print(f"saved {args.run / f'reading_{tag}.json'}")


if __name__ == "__main__":
    main()
