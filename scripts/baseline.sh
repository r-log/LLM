#!/usr/bin/env bash
# Run the full baseline on a GPU machine, unattended.
#
#   nohup bash scripts/baseline.sh > baseline.log 2>&1 &
#   tail -f baseline.log            # watch (Ctrl+C stops watching, not the job)
#
# Steps: install, tests, data download, speed test, then GPT-2-token and byte
# models (3 seeds each) at the same compute budget, then a summary table.
# Safe to re-run: finished steps (data, completed runs) are skipped.
#
# Settings (environment variables):
#   MINUTES_PER_RUN   target length of each training run (default 30)
#   MAX_FLOPS         upper limit on the budget per run (default 1e17)
#   PRESET            model size preset (default small)
#   SEEDS             default "1 2 3"
#   TOKENIZERS        default "gpt2 bytes"
#   DATA              default data/fineweb-edu (anything else must already be prepared)
#   COMPILE           default --compile (set to "" to turn off)
set -euo pipefail
cd "$(dirname "$0")/.."

MINUTES_PER_RUN=${MINUTES_PER_RUN:-30}
MAX_FLOPS=${MAX_FLOPS:-1e17}
PRESET=${PRESET:-small}
SEEDS=${SEEDS:-"1 2 3"}
TOKENIZERS=${TOKENIZERS:-"gpt2 bytes"}
DATA=${DATA:-data/fineweb-edu}
COMPILE=${COMPILE:---compile}
say() { echo; echo "=== $(date '+%H:%M:%S') $* ==="; }

say "1/5 install and test"
pip install -q -r requirements.txt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m pytest -q

say "2/5 data"
FIRST_TOK=${TOKENIZERS%% *}
if [ -f "$DATA/meta.json" ] && [ -f "$DATA/train_$FIRST_TOK.bin" ]; then
  echo "already prepared, skipping"
else
  python scripts/prepare_data.py --source fineweb-edu --train-bytes 3e9
fi

say "3/5 speed test"
python scripts/train.py --data "$DATA" --tokenizer "$FIRST_TOK" --preset "$PRESET" --steps 150 \
  --eval-every 100000 $COMPILE --no-save --name speedtest > /dev/null
FLOPS=$(python - "$MINUTES_PER_RUN" "$MAX_FLOPS" <<'EOF'
import json, sys
minutes, max_flops = float(sys.argv[1]), float(sys.argv[2])
cfg = json.load(open("runs/speedtest/config.json"))
recs = [json.loads(l) for l in open("runs/speedtest/log.jsonl")]
steady = [r["tok_per_s"] for r in recs if r["step"] >= 50 and "val_bpb" not in r]  # skip compile warm-up
tok_s = sorted(steady)[len(steady) // 2]
flops_s = tok_s * cfg["flops_per_step"] / cfg["tokens_per_step"]
budget = min(max_flops, flops_s * minutes * 60)
print(f"{tok_s:,.0f} tokens/s = {flops_s / 1e12:.1f} TFLOP/s achieved -> budget {budget:.2e} FLOPs per run", file=sys.stderr)
print(f"{budget:.3e}")
EOF
)
echo "using --flops $FLOPS (byte runs may take a bit longer or shorter than the target)"

say "4/5 training"
for s in $SEEDS; do
  for tok in $TOKENIZERS; do
    name="$tok-$PRESET-s$s"
    if [ -f "runs/$name/final.json" ]; then echo "$name done, skipping"; continue; fi
    say "train $name"
    python scripts/train.py --data "$DATA" --tokenizer "$tok" --preset "$PRESET" \
      --flops "$FLOPS" --seed "$s" $COMPILE --name "$name" --log-every 50
  done
done

say "5/5 summary"
python scripts/evaluate.py --summary runs/*-"$PRESET"-s*/ | tee "baseline_summary_$PRESET.txt"
for f in runs/*-"$PRESET"-s*/final.json; do
  python -c "import json,sys; r=json.load(open('$f')); print(f\"{'$f'.split('/')[1]:24s} time {r['train_time_s']/60:6.1f} min   bytes/token {r['clean']['bytes_per_token']:.2f}\")"
done
say "ALL DONE - you can stop the pod now (files in /workspace are kept)"
