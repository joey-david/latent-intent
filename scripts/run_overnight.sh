#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/qwen3_8b_counterfactual.yaml}"
SEEDS="${LATENT_INTENT_SEEDS:-20260607 20260818 20260819}"
LOG_DIR="${LATENT_INTENT_LOG_DIR:-logs/overnight}"
mkdir -p "$LOG_DIR"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

echo "Overnight Latent Intent replication suite"
echo "Config: $CONFIG"
echo "Seeds: $SEEDS"
echo "Logs: $LOG_DIR"

for seed in $SEEDS; do
  export RUN_SEED="$seed"
  export RUN_NAME="counterfactual-latent-intent-qwen3-8b-seed-${seed}"
  log="$LOG_DIR/${STAMP}-seed-${seed}.log"

  echo
  echo "=== seed $seed ==="
  echo "log: $log"
  CONFIG="$CONFIG" ./scripts/run_experiments.sh "$@" 2>&1 | tee "$log"
done

echo
echo "All overnight replications completed."
