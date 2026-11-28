"""Tokenizers.

Every tokenizer turns text into a list of integer ids and back. Two are provided:

- ByteTokenizer: one id per UTF-8 byte (vocab 256) plus one end-of-text id.
- GPT2Tokenizer: the standard GPT-2 BPE (vocab 50257) via tiktoken.
- BPETokenizer: a byte-level BPE trained with scripts/train_tokenizer.py, named
  "bpe-<name>" and loaded from data/tokenizers/<name>.json.

Both expose the same small interface, so the rest of the code never needs to
know which one it is using.
"""

from __future__ import annotations

from pathlib import Path

TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "data" / "tokenizers"


class ByteTokenizer:
    name = "bytes"

    def __init__(self) -> None:
        self.eot_id = 256  # one extra id after the 256 byte values
        self.vocab_size = 257

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def token_bytes(self, ids: list[int]) -> list[int]:
        """Number of text bytes each id stands for (0 for end-of-text)."""
        return [1 if i < 256 else 0 for i in ids]


class GPT2Tokenizer:
    name = "gpt2"

    def __init__(self) -> None:
        import tiktoken  # imported here so the byte path works without network

        self._enc = tiktoken.get_encoding("gpt2")
        self.eot_id = self._enc.eot_token
        self.vocab_size = self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    def token_bytes(self, ids: list[int]) -> list[int]:
        return [0 if i == self.eot_id else len(self._enc.decode_single_token_bytes(i)) for i in ids]


class BPETokenizer:
    """Byte-level BPE from the `tokenizers` library. Id 0 is end-of-text."""

    def __init__(self, name: str, path: Path | None = None) -> None:
        from tokenizers import Tokenizer

        self.name = name
        path = path or TOKENIZER_DIR / f"{name.removeprefix('bpe-')}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; train it with scripts/train_tokenizer.py")
        self._tok = Tokenizer.from_file(str(path))
        self.eot_id = self._tok.token_to_id("<|eot|>")
        self.vocab_size = self._tok.get_vocab_size()
        # Byte-level BPE writes every byte as one character, so a token's length in
        # characters is its length in bytes.
        self._nbytes = [0] * self.vocab_size
        for token, i in self._tok.get_vocab().items():
            self._nbytes[i] = 0 if i == self.eot_id else len(token)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode([i for i in ids if i != self.eot_id])

    def token_bytes(self, ids: list[int]) -> list[int]:
        return [self._nbytes[i] for i in ids]


def get_tokenizer(name: str):
    if name == "bytes":
        return ByteTokenizer()
    if name == "gpt2":
        return GPT2Tokenizer()
    if name.startswith("bpe-"):
        return BPETokenizer(name)
    raise ValueError(f"unknown tokenizer {name!r} (choose 'bytes', 'gpt2' or 'bpe-<name>')")
