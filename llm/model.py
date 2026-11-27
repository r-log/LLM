"""A small, standard GPT (close to nanoGPT, with common modern defaults):

- pre-norm with RMSNorm
- rotary position embeddings (RoPE) instead of learned positions
- GELU MLP, 4x wide
- input and output embeddings share weights ("tied")
- causal attention via PyTorch's fused scaled_dot_product_attention

Optional (off by default):
- local_window W: each position attends only to the last W positions (sliding window)
- mem_chunk k (needs local_window): positions older than the window are still visible,
  but compressed: every k consecutive positions are averaged into one key/value

Shapes in comments: B = batch, T = sequence length, C = model width,
H = number of heads, D = C // H (width per head).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 512  # longest sequence the model sees
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    local_window: int | None = None  # None = full causal attention
    mem_chunk: int = 0  # 0 = no compressed memory


def attention_layout(T: int, window: int, chunk: int, device) -> torch.Tensor:
    """Boolean mask (T, T + T//chunk): which keys each position may attend to.

    Columns 0..T-1 are the raw positions: visible if inside the last `window` positions.
    Columns T.. are compressed chunks (chunk j covers positions j*chunk .. j*chunk+chunk-1):
    visible once the chunk starts at least `window` positions back. Since chunk <= window,
    a visible chunk lies entirely in the past, so this stays causal.
    """
    t = torch.arange(T, device=device)[:, None]
    p = torch.arange(T, device=device)[None, :]
    raw = (p <= t) & (p > t - window)
    if not chunk:
        return raw
    j = torch.arange(T // chunk, device=device)[None, :]
    mem = j * chunk <= t - window
    return torch.cat([raw, mem], dim=1)


def avg_attended(T: int, window: int | None, chunk: int) -> float:
    """Average number of keys a position attends to (used for the FLOP count)."""
    if window is None:
        return T / 2
    total = 0
    for t in range(T):
        total += min(t + 1, window)
        if chunk and t >= window:
            total += min((t - window) // chunk + 1, T // chunk)
    return total / T


class RMSNorm(nn.Module):
    """Rescales each vector to unit size, then by a learned per-channel gain."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_cache(T: int, D: int, device, base: float = 10000.0):
    """Precompute the cos/sin tables RoPE uses to rotate query/key pairs."""
    freqs = 1.0 / (base ** (torch.arange(0, D, 2, device=device).float() / D))
    angles = torch.outer(torch.arange(T, device=device).float(), freqs)  # (T, D/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, D). Rotate each pair (x1, x2) by an angle that grows with position,
    # so the dot product between a query and a key depends on their distance.
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout
        self.window, self.chunk = cfg.local_window, cfg.mem_chunk
        if self.chunk:
            assert self.window and self.chunk <= self.window, "mem_chunk needs local_window >= mem_chunk"
        self._mask = {}

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        dp = self.dropout if self.training else 0.0
        if self.window is None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dp)
        else:
            key = (T, x.device)
            if key not in self._mask:
                self._mask[key] = attention_layout(T, self.window, self.chunk, x.device)
            mask = self._mask[key]
            if self.chunk:
                assert T % self.chunk == 0, f"sequence length {T} must be a multiple of mem_chunk {self.chunk}"
                M = T // self.chunk
                # Compress: average each chunk's (un-rotated) keys and values, then give the
                # chunk the position of its last byte for RoPE.
                k_mem = k.view(B, self.n_head, M, self.chunk, -1).mean(3)
                v_mem = v.view(B, self.n_head, M, self.chunk, -1).mean(3)
                last = torch.arange(M, device=x.device) * self.chunk + self.chunk - 1
                k_mem = apply_rope(k_mem, cos[last], sin[last])
                q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
                k, v = torch.cat([k, k_mem], dim=2), torch.cat([v, v_mem], dim=2)
            else:
                q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dp)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.n_embd), Attention(cfg)
        self.norm2, self.mlp = RMSNorm(cfg.n_embd), MLP(cfg)

    def forward(self, x, cos, sin):
        # Each sub-layer *adds* to the residual stream instead of replacing it.
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied embeddings
        self.apply(self._init)
        # Scale down the layers that write into the residual stream, so the
        # stream does not grow with depth at the start of training (GPT-2 trick).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))
        cos, sin = rope_cache(cfg.block_size, cfg.n_embd // cfg.n_head, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.wte.weight.numel() if non_embedding else n

    def flops_per_token(self, T: int | None = None) -> float:
        """Training FLOPs per token: 6 per weight (forward + backward), plus attention.

        Logged by every run so models can be compared at equal compute.
        Embedding lookups are nearly free and are left out; the output layer is not.
        """
        cfg = self.cfg
        T = T or cfg.block_size
        n = self.num_params(non_embedding=True) + cfg.vocab_size * cfg.n_embd  # + lm_head matmul
        # QK^T and attn*V, fwd+bwd: 12 * layers * width per attended key (PaLM formula).
        # Full causal attention attends to T/2 keys on average. With a local window or
        # compressed memory, we count the keys actually used by the design (masked-out
        # work in our simple implementation is not counted, as an efficient kernel skips it).
        return 6 * n + 12 * cfg.n_layer * cfg.n_embd * avg_attended(T, cfg.local_window, cfg.mem_chunk)

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. Returns logits (B, T, V), and the mean loss if targets given.

        Targets equal to -1 are ignored in the loss.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} longer than block_size {self.cfg.block_size}"
        x = self.wte(idx)
        cos, sin = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        return idx
