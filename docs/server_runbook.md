# Server Runbook

This repo is prepared so the server side only needs a token-filled `.env` and one
script invocation.

## Steps

```bash
cd latent-intent-probes
cp .env.example .env
$EDITOR .env
./scripts/run_experiments.sh
```

## Expected Runtime Behavior

- Creates `.venv/`.
- Installs the package and dependencies.
- Downloads the Hugging Face model into the configured HF cache.
- Fails fast if PyTorch cannot see CUDA, unless `ALLOW_CPU=true` is set.
- Writes a timestamped run directory under `results/`.
- Prints the final `summary.md` path when complete.
- Saves phase-wise activation probes, phase-transfer probes, and attention
  asymmetry summaries under `results/`.

## GPU Check

Before a full run:

```bash
./scripts/check_accel.sh
./scripts/check_accel.sh --check-model-placement
```

The first command checks torch and `nvidia-smi`. The second also loads the model
and prints the Transformers device map. Do not proceed if CUDA is unavailable or
the model is mapped to `cpu`/`disk`.

## Common Overrides

```bash
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
RUN_NAME=latent-intent-qwen15b
CONFIG=configs/default.yaml
```

The default run collects attention weights, so it uses `model.batch_size: 2`.
If VRAM is tight, lower `model.batch_size` in `configs/default.yaml`. If a model
does not support attention outputs with the configured implementation, set
`model.collect_attentions: false`; the phase activation battery will still run.

## Pullback

After the run, archive the results directory:

```bash
tar -czf latent-intent-results.tgz results/
```

Pull `latent-intent-results.tgz` back locally for analysis.
