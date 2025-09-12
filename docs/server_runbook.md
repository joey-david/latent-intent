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
- Writes a timestamped run directory under `results/`.
- Prints the final `summary.md` path when complete.

## Common Overrides

```bash
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
RUN_NAME=latent-intent-qwen15b
CONFIG=configs/default.yaml
```

If VRAM is tight, lower `model.batch_size` in `configs/default.yaml`.

## Pullback

After the run, archive the results directory:

```bash
tar -czf latent-intent-results.tgz results/
```

Pull `latent-intent-results.tgz` back locally for analysis.
