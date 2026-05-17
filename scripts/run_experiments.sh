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

CONFIG="${CONFIG:-configs/qwen3_8b_counterfactual.yaml}"
LOCAL_VENV="$ROOT_DIR/.venv"
DRY_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

# Respect an already-active environment. This is useful on shared GPU servers
# where a research venv already contains torch/transformers and rebuilding it is
# expensive. LATENT_INTENT_VENV can explicitly point at another environment.
if [[ -n "${LATENT_INTENT_VENV:-}" ]]; then
  VENV_DIR="$LATENT_INTENT_VENV"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  VENV_DIR="$VIRTUAL_ENV"
else
  VENV_DIR="$LOCAL_VENV"
fi

if [[ -x "$VENV_DIR/bin/python" ]]; then
  PYTHON="$VENV_DIR/bin/python"
elif (( DRY_RUN )); then
  # A dry run only needs Python + PyYAML. Do not create/install a heavyweight
  # model environment just to validate the paired dataset.
  PYTHON="$(command -v python3)"
else
  echo "Creating $VENV_DIR (first-time setup only) ..."
  python3 -m venv "$VENV_DIR"
  PYTHON="$VENV_DIR/bin/python"
fi

run_source() {
  PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$@"
}

if (( DRY_RUN )); then
  # Prefer the selected interpreter, but fall back to another existing Python
  # that already has PyYAML. No pip invocation is needed for --dry-run.
  if ! "$PYTHON" -c 'import yaml' >/dev/null 2>&1; then
    for candidate in "$LOCAL_VENV/bin/python" "$(command -v python3)"; do
      if [[ -x "$candidate" ]] && "$candidate" -c 'import yaml' >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
      fi
    done
  fi
  if ! "$PYTHON" -c 'import yaml' >/dev/null 2>&1; then
    echo "Dry-run needs PyYAML, but no existing Python environment with PyYAML was found." >&2
    echo "Activate your normal research venv first, then rerun this command." >&2
    exit 1
  fi

  echo "Dry run using: $PYTHON"
  echo "Running Latent Intent config: $CONFIG"
  run_source -m latent_intent_probe.run --config "$CONFIG" "$@"
  exit 0
fi

repair_pip() {
  if "$PYTHON" -m pip --version >/dev/null 2>&1; then
    return 0
  fi

  echo "pip is damaged in $VENV_DIR; repairing pip in place (keeping installed packages) ..."
  "$PYTHON" -m ensurepip --upgrade --default-pip >/dev/null 2>&1 || true
  if "$PYTHON" -m pip --version >/dev/null 2>&1; then
    return 0
  fi

  # If ensurepip saw the broken distribution as already installed, remove only
  # pip itself and restore the bundled copy. Never delete the whole environment.
  PURELIB="$($PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  rm -rf "$PURELIB/pip" "$PURELIB"/pip-*.dist-info
  "$PYTHON" -m ensurepip --upgrade --default-pip
  "$PYTHON" -m pip --version >/dev/null
}

repair_pip

# Run directly from src; no editable reinstall is needed when the repository
# changes. Install only dependencies that are actually absent from the selected
# environment, so an existing torch stack is reused untouched.
mapfile -t MISSING_DEPS < <("$PYTHON" - <<'PY'
import importlib.util

requirements = [
    ("accelerate", "accelerate>=0.33"),
    ("matplotlib", "matplotlib>=3.8"),
    ("numpy", "numpy>=1.26"),
    ("pandas", "pandas>=2.2"),
    ("dotenv", "python-dotenv>=1.0"),
    ("yaml", "pyyaml>=6.0"),
    ("sklearn", "scikit-learn>=1.4"),
    ("seaborn", "seaborn>=0.13"),
    ("tabulate", "tabulate>=0.9"),
    ("torch", "torch>=2.3"),
    ("tqdm", "tqdm>=4.66"),
    ("transformers", "transformers>=4.44,<5"),
]
for module, requirement in requirements:
    if importlib.util.find_spec(module) is None:
        print(requirement)
PY
)

if ((${#MISSING_DEPS[@]})); then
  echo "Installing only missing dependencies: ${MISSING_DEPS[*]}"
  "$PYTHON" -m pip install "${MISSING_DEPS[@]}"
else
  echo "Using existing environment; all experiment dependencies are already installed."
fi

echo "Python: $PYTHON"
echo "Running Latent Intent config: $CONFIG"
run_source -m latent_intent_probe.run --config "$CONFIG" "$@"
