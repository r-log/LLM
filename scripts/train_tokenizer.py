"""Train a byte-level BPE tokenizer and save it as data/tokenizers/<name>.json.

    python scripts/train_tokenizer.py --source code --vocab 4096 --name code4k
    python scripts/train_tokenizer.py --source code --vocab 16384 --name code16k

Then use it everywhere as tokenizer "bpe-<name>", e.g. prepare_data.py --tokenizers bpe-code4k.
Byte-level means every possible byte is in the vocabulary, so any text round-trips exactly.
Id 0 is reserved for end-of-text. The vocabulary must stay below 65536 (ids are stored as uint16).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm.tokenizers import TOKENIZER_DIR, BPETokenizer  # noqa: E402


def texts(source: str, args):
    import prepare_data as pd  # the same document rendering as training

    if source == "code":
        it = pd.code_docs("train", args.code_raw)
    elif source == "stdlib":
        it = pd.stdlib_docs()
    else:
        raise ValueError(source)
    total = 0
    for doc in it:
        yield doc
        total += len(doc.encode("utf-8"))
        if total >= args.max_bytes:
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["code", "stdlib"], required=True)
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-bytes", type=float, default=3e8, help="train on at most this much text")
    ap.add_argument("--code-raw", default="data/code-raw")
    args = ap.parse_args()
    assert 256 < args.vocab < 65536

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab, special_tokens=["<|eot|>"], min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts(args.source, args), trainer)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    path = TOKENIZER_DIR / f"{args.name}.json"
    tok.save(str(path))

    # Check: exact round trip and bytes per token on a sample.
    t = BPETokenizer(f"bpe-{args.name}")
    sample = next(texts(args.source, args))[:200_000]
    ids = t.encode(sample)
    assert t.decode(ids) == sample, "round trip failed"
    assert sum(t.token_bytes(ids)) == len(sample.encode("utf-8"))
    print(f"saved {path}: vocab {t.vocab_size}, {len(sample.encode()) / len(ids):.2f} bytes per token on a sample")


if __name__ == "__main__":
    main()
