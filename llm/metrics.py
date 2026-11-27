"""Bits per byte: a tokenizer-independent quality score.

Why not "loss per token"? Because a token is a different amount of text for
every tokenizer. A byte model's tokens are 1 byte; GPT-2's tokens average ~4.5
bytes of English. Comparing their per-token losses is like comparing prices per
package without checking what's in the package.

Bits per byte asks one tokenizer-independent question: how many bits does the
model need, on average, to encode each byte of this exact text?

    bpb = (total loss over the text, in nats) / (ln 2 * number of UTF-8 bytes)

The loss is summed (not averaged) over every predicted token, converted from
nats to bits by dividing by ln 2, and divided by the byte length of the original
text. Lower is better. As a rough guide: a random guess over bytes is 8 bpb,
small models on web text land around 1.0-1.3, strong large models below 0.7.

How the text is fed in: documents are joined with an end-of-text token in
front of each one and cut into windows of `block_size`. Every token of every
document is predicted exactly once; the end-of-text tokens are only used as
context and are never scored, so the score covers exactly the document bytes.
The first tokens of each window have less context than they would with
overlapping windows, which makes scores slightly worse but identical in method
for every model, which is what matters.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def build_eval_stream(docs: list[str], tokenizer) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Returns (ids, score_mask, n_bytes) for a list of documents.

    ids[i] is the input token; score_mask[i] says whether ids[i] is a document
    token that should be predicted (True) or an end-of-text separator (False).
    """
    ids, mask, n_bytes = [], [], 0
    for doc in docs:
        toks = tokenizer.encode(doc)
        ids += [tokenizer.eot_id] + toks
        mask += [False] + [True] * len(toks)
        n_bytes += len(doc.encode("utf-8"))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool), n_bytes


@torch.no_grad()
def per_token_nats(model, docs: list[str], tokenizer, device="cpu", batch_size: int = 16):
    """The loss (in nats) of every document token, in text order.

    Returns (nats, window_pos, n_bytes): nats[i] is the loss of the i-th document token,
    window_pos[i] is how many positions of context it had inside its window.
    """
    was_training = model.training
    model.eval()
    ids, mask, n_bytes = build_eval_stream(docs, tokenizer)
    T = model.cfg.block_size
    # Inputs are ids[s : s+T], targets are ids[s+1 : s+T+1]. Windows don't overlap.
    starts = list(range(0, len(ids) - 1, T))
    out_nats, out_pos = [], []
    for b in range(0, len(starts), batch_size):
        xs, ys = [], []
        for s in starts[b : b + batch_size]:
            x = ids[s : s + T]
            y = ids[s + 1 : s + T + 1].clone()
            y[~mask[s + 1 : s + T + 1]] = -1  # never score separators
            pad = T - len(x)
            if pad:  # last window: pad the input, ignore the padded targets
                x = F.pad(x, (0, pad), value=tokenizer.eot_id)
            if len(y) < T:
                y = F.pad(y, (0, T - len(y)), value=-1)
            xs.append(x)
            ys.append(y)
        x, y = torch.stack(xs).to(device), torch.stack(ys).to(device)
        with _autocast(device):
            logits, _ = model(x)
        nll = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1),
                              ignore_index=-1, reduction="none").view(y.shape)
        keep = y != -1
        pos = torch.arange(T, device=y.device).expand_as(y)
        out_nats.append(nll[keep].cpu())  # row-major order = text order
        out_pos.append(pos[keep].cpu())
    if was_training:
        model.train()
    nats, window_pos = torch.cat(out_nats), torch.cat(out_pos)
    # Sanity check: every document token was scored exactly once.
    assert len(nats) == int(mask.sum()), (len(nats), int(mask.sum()))
    return nats, window_pos, n_bytes


def bits_per_byte(model, docs: list[str], tokenizer, device="cpu", batch_size: int = 16) -> dict:
    """Score `model` on `docs`. Returns bits per byte plus the raw totals behind it."""
    nats, _, n_bytes = per_token_nats(model, docs, tokenizer, device, batch_size)
    total_nats, n_scored = nats.double().sum().item(), len(nats)
    return {
        "bpb": total_nats / (math.log(2) * n_bytes),
        "nats": total_nats,
        "bytes": n_bytes,
        "tokens": n_scored,
        "bytes_per_token": n_bytes / max(n_scored, 1),
    }


def _autocast(device):
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)
