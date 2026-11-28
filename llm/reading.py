"""Reading curves: how well a model predicts a document (e.g. a whole repository) as it
reads further into it, optionally learning while it reads.

The document is processed in windows of the model's block size T that start every S
positions (S = T: no overlap; S < T: every token after the first window has at least T - S
positions of context). Each window is first scored, then, when adapting, the model takes
an optimizer step on (a selection of) the tokens it just scored. Scoring always happens
before the update, so a token is never predicted by a model that has trained on it.

Selection rules for which scored tokens to learn from:
    all        every scored token
    surprise   the `frac` most surprising tokens (highest loss) of the window
    random     a random `frac` of the window's tokens (the control for `surprise`)
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

BINS = [0, 4_096, 16_384, 65_536, 262_144, 1_048_576, float("inf")]


def bin_label(lo, hi) -> str:
    def h(x):
        return "end" if x == float("inf") else f"{int(x) // 1024}K"
    return f"{h(lo)}-{h(hi)}"


def params_to_update(model, spec: str):
    """'all', 'blocks:N' (the last N transformer blocks) or 'mlp' (all MLP layers)."""
    if spec == "all":
        return list(model.parameters())
    if spec.startswith("blocks:"):
        n = int(spec.split(":")[1])
        return [p for b in model.blocks[-n:] for p in b.parameters()]
    if spec == "mlp":
        return [p for b in model.blocks for p in b.mlp.parameters()]
    raise ValueError(f"unknown parameter set {spec!r}")


def _autocast(device):
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def read_document(model, ids: list[int], *, stride: int | None = None, select: str | None = None,
                  frac: float = 1.0, lr: float = 0.0, opt: str = "adam", params: str = "all",
                  steps: int = 1, device="cpu", seed: int = 0, pad_id: int = 0):
    """Score one document (ids include a leading end-of-text) and optionally learn while reading.

    Returns (nats, updates): nats[i] is the loss of predicting ids[i + 1]; updates counts
    optimizer steps taken. The model is modified in place when lr > 0; pass a copy.
    """
    T = model.cfg.block_size
    S = stride or T
    n = len(ids)
    ids_t = torch.tensor(ids, dtype=torch.long)
    nats = torch.zeros(n - 1)
    adapt = select is not None and lr > 0
    if adapt:
        ps = params_to_update(model, params)
        optimizer = (torch.optim.Adam(ps, lr=lr, betas=(0.9, 0.999)) if opt == "adam"
                     else torch.optim.SGD(ps, lr=lr))
        gen = torch.Generator().manual_seed(seed)
    model.eval()  # no dropout; gradients are still computed when adapting
    updates = 0
    starts = [0] + list(range(S, max(n - 1 - (T - S), S), S))
    for s in starts:
        y = ids_t[s + 1 : s + T + 1]
        L = len(y)
        lo = 0 if s == 0 else T - S  # first scored position inside this window
        if L <= lo:
            continue
        x = ids_t[s : s + L]
        if L < T:  # pad to the block size (needed by windowed/memory models), ignore the padding
            x = F.pad(x, (0, T - L), value=pad_id)
        x = x[None].to(device)
        yd = y.to(device)
        with torch.set_grad_enabled(adapt), _autocast(device):
            logits, _ = model(x)
        nll = F.cross_entropy(logits[0, :L].float(), yd, reduction="none")
        nats[s + lo : s + L] = nll[lo:L].detach().cpu()
        if not adapt:
            continue
        scored = torch.arange(lo, L)
        if select == "all":
            chosen = scored
        else:
            k = max(1, round(frac * len(scored)))
            if select == "surprise":
                chosen = scored[nll[lo:L].detach().cpu().topk(k).indices]
            elif select == "random":
                chosen = scored[torch.randperm(len(scored), generator=gen)[:k]]
            else:
                raise ValueError(f"unknown selection rule {select!r}")
        for step in range(steps):
            if step > 0:  # extra steps need a fresh forward pass
                with _autocast(device):
                    logits, _ = model(x)
                nll = F.cross_entropy(logits[0, :L].float(), yd, reduction="none")
            loss = nll[chosen.to(device)].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
    return nats, updates


def curve(nats: torch.Tensor, ids: list[int], tokenizer) -> dict:
    """Bits per byte in bins of 'bytes of the document already read'."""
    nbytes = torch.tensor(tokenizer.token_bytes(ids[1:]), dtype=torch.float64)
    seen = torch.cumsum(nbytes, 0) - nbytes  # bytes read before each predicted token
    out = {}
    for lo, hi in zip(BINS, BINS[1:]):
        m = (seen >= lo) & (seen < hi)
        if nbytes[m].sum() > 0:
            out[bin_label(lo, hi)] = {"nats": float(nats[m].double().sum()), "bytes": float(nbytes[m].sum())}
    out["total"] = {"nats": float(nats.double().sum()), "bytes": float(nbytes.sum())}
    return out


def bpb(entry: dict) -> float:
    return entry["nats"] / (math.log(2) * entry["bytes"])


def run(model, docs: list[str], tokenizer, *, forget_docs: list[str] | None = None, **kw) -> dict:
    """Reading curve over several documents. The model is reset before each document.
    With forget_docs, also measures bits per byte on them before and after each adaptation."""
    base_state = copy.deepcopy(model.state_dict())
    totals, per_doc, updates, forget = {}, [], 0, []

    def score(texts):
        n = b = 0.0
        for t in texts:
            ids = [tokenizer.eot_id] + tokenizer.encode(t)
            nats, _ = read_document(model, ids, device=kw.get("device", "cpu"), pad_id=tokenizer.eot_id)
            n += float(nats.double().sum())
            b += len(t.encode("utf-8"))
        return n / (math.log(2) * b)

    forget_before = score(forget_docs) if forget_docs else None
    base_seed = kw.pop("seed", 0)
    for i, doc in enumerate(docs):
        model.load_state_dict(base_state)
        ids = [tokenizer.eot_id] + tokenizer.encode(doc)
        nats, u = read_document(model, ids, seed=base_seed + i, pad_id=tokenizer.eot_id, **kw)
        updates += u
        c = curve(nats, ids, tokenizer)
        per_doc.append(bpb(c["total"]))
        for k, v in c.items():
            t = totals.setdefault(k, {"nats": 0.0, "bytes": 0.0})
            t["nats"] += v["nats"]
            t["bytes"] += v["bytes"]
        if forget_docs:
            forget.append(score(forget_docs) - forget_before)
    model.load_state_dict(base_state)
    result = {"curve": {k: bpb(v) for k, v in totals.items()}, "per_doc_bpb": per_doc, "updates": updates}
    if forget_docs:
        result["forgetting"] = {"before_bpb": forget_before, "mean_increase": sum(forget) / len(forget),
                                "per_doc_increase": forget}
    return result
