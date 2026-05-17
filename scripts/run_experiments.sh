#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

SETUP_STAMP=".venv/.latent_intent_setup.ok"
if [[ "${FORCE_SETUP:-0}" == "1" || ! -f "$SETUP_STAMP" || pyproject.toml -nt "$SETUP_STAMP" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -e .
  touch "$SETUP_STAMP"
else
  echo "Using existing .venv setup; set FORCE_SETUP=1 to reinstall dependencies."
fi

CONFIG="${CONFIG:-configs/default.yaml}"
python -m latent_intent_probe.run --config "$CONFIG" "$@"
