# llm

A small GPT training and evaluation setup.

```
llm/
  tokenizers.py   bytes (vocab 257) and GPT-2 BPE (vocab 50257), same interface
  model.py        a small, standard GPT
  metrics.py      bits per byte (comparable across tokenizers)
  typos.py        deterministic typo noise
  evaluation.py   evaluation suite: clean + typos at 3% and 10%
scripts/
  prepare_data.py download text -> token files + fixed validation set
  train.py        train with a compute budget (--flops) or --steps, then evaluate
  evaluate.py     compare runs; mean ± std over seeds
tests/
```

## Quick start (CPU, offline)

```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/prepare_data.py --source stdlib
python scripts/train.py --data data/stdlib --tokenizer bytes --preset tiny --steps 300
```

## GPU

```bash
pip install -r requirements.txt
python scripts/prepare_data.py --source fineweb-edu --train-bytes 3e9
for s in 1 2 3; do
  python scripts/train.py --data data/fineweb-edu --tokenizer gpt2  --preset small --flops 1e17 --seed $s --compile --name gpt2-small-s$s
  python scripts/train.py --data data/fineweb-edu --tokenizer bytes --preset small --flops 1e17 --seed $s --compile --name bytes-small-s$s
done
python scripts/evaluate.py --summary runs/*-small-s*/
```
