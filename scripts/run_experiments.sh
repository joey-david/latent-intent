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

VENV_DIR=".venv"

create_venv() {
  echo "Creating a clean virtual environment in $VENV_DIR ..."
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
}

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  create_venv
elif ! "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "Existing $VENV_DIR has a broken pip installation; rebuilding it."
  create_venv
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# A newly-created venv should contain pip via ensurepip, but repair it once if
# the platform's venv bootstrap produced an incomplete installation.
if ! python -m pip --version >/dev/null 2>&1; then
  echo "Repairing pip with ensurepip ..."
  python -m ensurepip --upgrade
fi

SETUP_STAMP="$VENV_DIR/.latent_intent_setup.ok"
if [[ "${FORCE_SETUP:-0}" == "1" || ! -f "$SETUP_STAMP" || pyproject.toml -nt "$SETUP_STAMP" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -e .
  touch "$SETUP_STAMP"
else
  echo "Using existing .venv setup; set FORCE_SETUP=1 to reinstall dependencies."
fi

CONFIG="${CONFIG:-configs/qwen3_8b_counterfactual.yaml}"
echo "Running Latent Intent config: $CONFIG"
python -m latent_intent_probe.run --config "$CONFIG" "$@"
