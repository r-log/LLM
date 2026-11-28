#!/usr/bin/env bash
# Build all code data in one go (network + CPU; no GPU needed). Safe to re-run: finished
# steps are skipped.
#
#   nohup bash scripts/prepare_code_data.sh > data_build.log 2>&1 &
#
# 1. C-family code corpus from GitHub      -> data/code-raw/
# 2. C-family documentation corpus         -> data/docs-raw/
# 3. BPE tokenizers trained on the code    -> data/tokenizers/code4k.json, code16k.json
# 4. Training files: code only, and code + documentation
#                                          -> data/code/, data/code+docs/
#
# Settings (environment variables):
#   SCALE         multiply the number of repositories per language (default 1, ~500 repos)
#   DOCS_ARGS     extra build_docs_corpus.py arguments, e.g. "--allow-research-only"
#   DOCS_REPEAT   how many times the documentation is mixed in (default 1)
#   TOKENIZERS    default "bytes,bpe-code4k,bpe-code16k"
#   GITHUB_TOKEN  optional; raises GitHub's search rate limit
set -euo pipefail
cd "$(dirname "$0")/.."
SCALE=${SCALE:-1}
DOCS_ARGS=${DOCS_ARGS:-}
DOCS_REPEAT=${DOCS_REPEAT:-1}
TOKENIZERS=${TOKENIZERS:-bytes,bpe-code4k,bpe-code16k}
say() { echo; echo "=== $(date '+%H:%M:%S') $* ==="; }

pip install -q -r requirements.txt

say "1/4 code corpus"
if [ -f data/code-raw/train.jsonl ]; then echo "exists, skipping"; else
  [ -f data/code-raw/repos.json ] || python scripts/build_code_corpus.py search --scale "$SCALE"
  python scripts/build_code_corpus.py download
  python scripts/build_code_corpus.py extract
fi

say "2/4 documentation corpus"
if [ -f data/docs-raw/docs.jsonl ]; then echo "exists, skipping"; else
  python scripts/build_docs_corpus.py $DOCS_ARGS
fi

say "3/4 tokenizers"
for v in 4096:code4k 16384:code16k; do
  name=${v#*:}
  if [ -f "data/tokenizers/$name.json" ]; then echo "$name exists, skipping"; else
    python scripts/train_tokenizer.py --source code --vocab "${v%%:*}" --name "$name"
  fi
done

say "4/4 training files"
[ -f data/code/meta.json ] || python scripts/prepare_data.py --source code --tokenizers "$TOKENIZERS"
[ -f data/code+docs/meta.json ] || python scripts/prepare_data.py --source code --tokenizers "$TOKENIZERS" \
  --add-docs data/docs-raw/docs.jsonl --docs-repeat "$DOCS_REPEAT"

say "summary"
cat data/code-raw/summary.json data/docs-raw/summary.json
python - <<'PY'
import json
for d in ("data/code", "data/code+docs"):
    m = json.load(open(f"{d}/meta.json"))
    print(d, {k: m[k] for k in ("val_docs", "val_bytes", "train_docs", "train_bytes", "train_docs_bytes", "bytes_per_token")})
PY
say "DATA DONE"
