"""Fast checks that the measuring tools are right. Run: python -m pytest -q"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.metrics import bits_per_byte, build_eval_stream  # noqa: E402
from llm.model import GPT, GPTConfig  # noqa: E402
from llm.tokenizers import ByteTokenizer  # noqa: E402
from llm.typos import add_typos  # noqa: E402

DOCS = ["Hello world, this is a test.", "Zweite Größe: ünïcödé bytes count too.", "x" * 300]


class Uniform(torch.nn.Module):
    """A 'model' that gives every id equal probability: its bpb is known exactly."""

    def __init__(self, vocab, block):
        super().__init__()
        self.cfg = GPTConfig(vocab_size=vocab, block_size=block)

    def forward(self, idx, targets=None):
        return torch.zeros(*idx.shape, self.cfg.vocab_size), None


def test_uniform_byte_model_bpb_is_log2_vocab():
    # Each byte is one token, each costs ln(257) nats -> log2(257) bits per byte.
    # Holds for UTF-8 too (non-ASCII letters are 2 bytes = 2 tokens).
    tok = ByteTokenizer()
    r = bits_per_byte(Uniform(257, 64), DOCS, tok)
    assert math.isclose(r["bpb"], math.log2(257), rel_tol=1e-6)
    assert r["bytes"] == sum(len(d.encode()) for d in DOCS)


def test_every_token_scored_once_regardless_of_window():
    tok = ByteTokenizer()
    for block in (7, 64, 1000):
        r = bits_per_byte(Uniform(257, block), DOCS, tok)
        assert r["tokens"] == sum(len(d.encode()) for d in DOCS)


def test_eval_stream_masks_separators():
    ids, mask, n = build_eval_stream(["ab", "c"], ByteTokenizer())
    assert ids.tolist() == [256, 97, 98, 256, 99]
    assert mask.tolist() == [False, True, True, False, True]
    assert n == 3


def test_typos_deterministic_and_rate():
    text = "the quick brown fox jumps over the lazy dog " * 200
    a, b = add_typos(text, 0.1, seed=1), add_typos(text, 0.1, seed=1)
    assert a == b and a != text
    assert add_typos(text, 0.0, seed=1) == text
    changed = sum(x != y for x, y in zip(text, a))
    assert changed > 0.03 * len(text)  # rough: corruption actually happens


def test_model_forward_and_loss_near_log_vocab_at_init():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=257, block_size=32, n_layer=2, n_head=2, n_embd=64)
    m = GPT(cfg)
    x = torch.randint(0, 257, (4, 32))
    logits, loss = m(x, torch.randint(0, 257, (4, 32)))
    assert logits.shape == (4, 32, 257)
    assert abs(loss.item() - math.log(257)) < 0.5  # untrained ≈ uniform guess


def test_model_is_causal():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=257, block_size=16, n_layer=2, n_head=2, n_embd=32)).eval()
    x = torch.randint(0, 257, (1, 16))
    y = x.clone()
    y[0, 10:] = (y[0, 10:] + 1) % 257  # change the future
    a, _ = m(x)
    b, _ = m(y)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-5)  # the past didn't see it


def _small(**kw):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=257, block_size=64, n_layer=2, n_head=2, n_embd=32, **kw)).eval()


def test_standard_flops_unchanged():
    # The baseline budgets depend on this number; it must never drift.
    m = GPT(GPTConfig(vocab_size=257, block_size=512, n_layer=6, n_head=6, n_embd=384))
    assert abs(m.flops_per_token() - 7.140e7) / 7.140e7 < 1e-3


def test_full_window_equals_standard_attention():
    a, b = _small(), _small(local_window=64)
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 257, (2, 64))
    assert torch.allclose(a(x)[0], b(x)[0], atol=1e-5)


def test_window_and_memory_are_causal():
    for kw in ({"local_window": 16}, {"local_window": 16, "mem_chunk": 4}):
        m = _small(**kw)
        x = torch.randint(0, 257, (1, 64))
        y = x.clone()
        y[0, 40:] = (y[0, 40:] + 1) % 257
        assert torch.allclose(m(x)[0][0, :40], m(y)[0][0, :40], atol=1e-5), kw


def test_window_forgets_and_memory_remembers():
    # Change byte 2. At position 60 it is 58 positions back: outside a 16-wide window.
    x = torch.randint(0, 257, (1, 64))
    y = x.clone()
    y[0, 2] = (y[0, 2] + 1) % 257
    win = _small(local_window=16)
    assert torch.allclose(win(x)[0][0, 60], win(y)[0][0, 60], atol=1e-5)  # window alone: forgotten
    mem = _small(local_window=16, mem_chunk=4)
    assert not torch.allclose(mem(x)[0][0, 60], mem(y)[0][0, 60], atol=1e-5)  # memory: still felt
    # No gap: at t=18 byte 2 has just left the raw window, and its chunk (0-3) is already visible.
    # One layer only: with more layers, information also hops forward window by window.
    torch.manual_seed(0)
    win1 = GPT(GPTConfig(vocab_size=257, block_size=64, n_layer=1, n_head=2, n_embd=32, local_window=16)).eval()
    mem1 = GPT(GPTConfig(vocab_size=257, block_size=64, n_layer=1, n_head=2, n_embd=32, local_window=16, mem_chunk=4)).eval()
    assert torch.allclose(win1(x)[0][0, 18], win1(y)[0][0, 18], atol=1e-5)
    assert not torch.allclose(mem1(x)[0][0, 18], mem1(y)[0][0, 18], atol=1e-5)


def test_flops_ordering():
    def f(**kw):
        return GPT(GPTConfig(vocab_size=257, block_size=2048, n_layer=6, n_head=6, n_embd=384, **kw)).flops_per_token()
    full, win, mem = f(), f(local_window=512), f(local_window=512, mem_chunk=4)
    assert win < mem < full
