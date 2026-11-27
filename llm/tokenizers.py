"""Tokenizers.

Every tokenizer turns text into a list of integer ids and back. Two are provided:

- ByteTokenizer: one id per UTF-8 byte (vocab 256) plus one end-of-text id.
- GPT2Tokenizer: the standard GPT-2 BPE (vocab 50257) via tiktoken.

Both expose the same small interface, so the rest of the code never needs to
know which one it is using.
"""

from __future__ import annotations


class ByteTokenizer:
    name = "bytes"

    def __init__(self) -> None:
        self.eot_id = 256  # one extra id after the 256 byte values
        self.vocab_size = 257

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


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


def get_tokenizer(name: str):
    if name == "bytes":
        return ByteTokenizer()
    if name == "gpt2":
        return GPT2Tokenizer()
    raise ValueError(f"unknown tokenizer {name!r} (choose 'bytes' or 'gpt2')")
