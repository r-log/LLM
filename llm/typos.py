"""Deterministic typo noise, for the robustness test set.

Same text + same seed + same rate always gives the same corrupted text, so every
model is tested on exactly the same typos.

Each letter is corrupted with probability `rate`, using one of five edits that
match how people actually mistype:

    delete      "hello" -> "helo"
    duplicate   "hello" -> "helllo"
    swap        "hello" -> "hlelo"     (with the next character)
    neighbor    "hello" -> "hwllo"     (a key next to it on a QWERTY keyboard)
    case        "hello" -> "hEllo"
"""

from __future__ import annotations

import random

_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def _build_neighbors() -> dict[str, str]:
    pos = {c: (r, i) for r, row in enumerate(_ROWS) for i, c in enumerate(row)}
    out = {}
    for c, (r, i) in pos.items():
        out[c] = "".join(
            d for d, (r2, i2) in pos.items() if d != c and abs(r - r2) <= 1 and abs(i - i2) <= 1
        )
    return out


_NEIGHBORS = _build_neighbors()
EDITS = ("delete", "duplicate", "swap", "neighbor", "case")


def add_typos(text: str, rate: float, seed: int) -> str:
    rng = random.Random(seed)
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if not c.isalpha() or rng.random() >= rate:
            out.append(c)
            i += 1
            continue
        edit = rng.choice(EDITS)
        if edit == "delete":
            pass
        elif edit == "duplicate":
            out += [c, c]
        elif edit == "swap" and i + 1 < len(chars):
            out += [chars[i + 1], c]
            i += 1  # the next character was consumed by the swap
        elif edit == "neighbor" and c.lower() in _NEIGHBORS:
            n = rng.choice(_NEIGHBORS[c.lower()])
            out.append(n.upper() if c.isupper() else n)
        elif edit == "case":
            out.append(c.swapcase())
        else:  # edit not possible here (e.g. swap at the end); keep the letter
            out.append(c)
        i += 1
    return "".join(out)
