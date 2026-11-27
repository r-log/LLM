"""Train one model and evaluate it.

Each run writes to runs/<name>/:
    config.json   every setting, so the run can be repeated exactly
    log.jsonl     one line per log step (loss, learning rate, compute used, speed)
    final.json    the full evaluation suite at the end
    model.pt      the trained weights (skip with --no-save)

The training length is set by a compute budget, not a step count, so a byte
model and a token model can be compared fairly: --flops 1e16 means "spend this
many floating point operations", and the script works out the steps.
(--steps still works, for quick tests.)

Examples:
    # 1-2 minute CPU smoke test
    python scripts/train.py --data data/stdlib --tokenizer bytes --preset tiny --steps 200

    # baseline on a GPU, one of three seeds
    python scripts/train.py --data data/fineweb-edu --tokenizer gpt2 --preset small \\
        --flops 1e17 --seed 1 --name baseline-gpt2-small-s1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.evaluation import load_val_docs, run_suite, summarize  # noqa: E402
from llm.metrics import bits_per_byte  # noqa: E402
from llm.model import GPT, GPTConfig  # noqa: E402
from llm.tokenizers import get_tokenizer  # noqa: E402

PRESETS = {
    # name: (n_layer, n_head, n_embd, block_size)
    "tiny": (2, 2, 128, 128),  # CPU smoke tests only
    "small": (6, 6, 384, 512),  # ~10M non-embedding params: default
    "medium": (8, 8, 512, 512),  # ~25M: for "does it still win when bigger?"
    "large": (12, 12, 768, 1024),  # ~85M (GPT-2 small shape)
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder made by prepare_data.py")
    ap.add_argument("--tokenizer", choices=["bytes", "gpt2"], required=True)
    ap.add_argument("--preset", choices=PRESETS, default="small")
    ap.add_argument("--block-size", type=int, help="override the preset's context length")
    budget = ap.add_mutually_exclusive_group(required=True)
    budget.add_argument("--flops", type=float, help="training compute budget")
    budget.add_argument("--steps", type=int, help="fixed number of optimizer steps")
    ap.add_argument("--batch-size", type=int, default=32, help="sequences per micro-batch")
    ap.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr-frac", type=float, default=0.1, help="final lr as a fraction of --lr")
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0, help="steps between quick val checks (0 = 10 per run)")
    ap.add_argument("--eval-docs", type=int, default=200, help="docs used for the quick in-training check")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (GPU)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--name", help="run name (default: built from the settings)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--no-save", action="store_true")
    return ap.parse_args()


class TrainData:
    """Samples random windows from the token file. Seeded, so runs are repeatable."""

    def __init__(self, path: Path, block_size: int, batch_size: int, seed: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.T, self.B = block_size, batch_size
        self.rng = np.random.default_rng(seed)
        if len(self.data) <= block_size + 1:
            raise ValueError(f"{path} has only {len(self.data)} tokens; need more than block_size {block_size}")

    def batch(self, device):
        starts = self.rng.integers(0, len(self.data) - self.T - 1, size=self.B)
        chunk = np.stack([self.data[s : s + self.T + 1].astype(np.int64) for s in starts])
        chunk = torch.from_numpy(chunk)
        x, y = chunk[:, :-1], chunk[:, 1:]
        if str(device).startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


def lr_at(step, total, args):
    """Linear warmup, then cosine decay down to min_lr_frac * lr."""
    warmup = max(1, int(args.warmup_frac * total))
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device
    tok = get_tokenizer(args.tokenizer)
    n_layer, n_head, n_embd, block = PRESETS[args.preset]
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size or block,
                    n_layer=n_layer, n_head=n_head, n_embd=n_embd)
    model = GPT(cfg).to(device)

    tokens_per_step = args.batch_size * args.grad_accum * cfg.block_size
    flops_per_step = model.flops_per_token() * tokens_per_step
    total_steps = args.steps or max(1, int(args.flops / flops_per_step))
    eval_every = args.eval_every or max(1, total_steps // 10)

    name = args.name or f"{args.tokenizer}-{args.preset}-s{args.seed}"
    run_dir = Path(args.runs_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(args.data) / "meta.json").read_text())
    config = {"args": vars(args), "model": asdict(cfg), "params_non_embedding": model.num_params(),
              "params_total": model.num_params(non_embedding=False), "total_steps": total_steps,
              "tokens_per_step": tokens_per_step, "flops_per_step": flops_per_step,
              "planned_flops": flops_per_step * total_steps, "data_meta": meta}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    bpt = meta["bytes_per_token"][args.tokenizer]
    print(f"run {name}: {model.num_params() / 1e6:.2f}M non-embedding params, {total_steps} steps, "
          f"{tokens_per_step * total_steps / 1e6:.1f}M tokens (~{tokens_per_step * total_steps * bpt / 1e6:.1f}MB of text), "
          f"{flops_per_step * total_steps:.2e} FLOPs, device {device}")

    data = TrainData(Path(args.data) / f"train_{args.tokenizer}.bin", cfg.block_size, args.batch_size, args.seed)
    val_docs = load_val_docs(args.data)
    quick_docs = val_docs[: args.eval_docs]

    # Weight decay only on matrices, not on norm gains (standard practice).
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=str(device).startswith("cuda"))
    fwd = torch.compile(model) if args.compile else model
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else torch.autocast("cpu", enabled=False)

    log = open(run_dir / "log.jsonl", "w")
    t0 = time.time()
    t_last, last_logged = t0, -1
    for step in range(total_steps):
        lr = lr_at(step, total_steps, args)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_sum = 0.0
        for _ in range(args.grad_accum):
            x, y = data.batch(device)
            with amp:
                _, loss = fwd(x, y)
            (loss / args.grad_accum).backward()
            loss_sum += loss.item() / args.grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        opt.step()
        opt.zero_grad(set_to_none=True)

        last = step == total_steps - 1
        do_eval = (step > 0 and step % eval_every == 0) or last
        if step % args.log_every == 0 or do_eval:
            now = time.time()
            rec = {"step": step, "loss": loss_sum, "train_bpb_est": loss_sum / math.log(2) / bpt,
                   "lr": lr, "grad_norm": grad_norm, "flops": flops_per_step * (step + 1),
                   "tok_per_s": tokens_per_step * (step - last_logged) / max(now - t_last, 1e-9),
                   "elapsed_s": now - t0}
            if do_eval:
                rec["val_bpb"] = bits_per_byte(model, quick_docs, tok, device)["bpb"]
            t_last, last_logged = time.time(), step  # eval time doesn't count against speed
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {step:5d} | loss {loss_sum:.4f} | ~bpb {rec['train_bpb_est']:.3f}"
                  + (f" | val bpb {rec['val_bpb']:.4f}" if "val_bpb" in rec else "")
                  + f" | lr {lr:.2e} | {rec['tok_per_s']:,.0f} tok/s", flush=True)
    log.close()

    print("final evaluation on all validation docs ...")
    results = run_suite(model, tok, val_docs, device)
    results["train_time_s"] = time.time() - t0
    (run_dir / "final.json").write_text(json.dumps(results, indent=2))
    print(summarize(results))
    if not args.no_save:
        torch.save({"model": model.state_dict(), "config": asdict(cfg), "tokenizer": args.tokenizer}, run_dir / "model.pt")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
