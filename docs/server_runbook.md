# Server Runbook

The default experiment is ready to run with one script invocation.

```bash
cd latent-intent
git pull
./scripts/run_experiments.sh
```

The script creates `.venv`, installs the package, loads `Qwen/Qwen3-8B`, and writes a timestamped run under `results/`.

Before the full run, two useful checks are:

```bash
# Build and inspect 8 matched pairs without loading the model.
./scripts/run_experiments.sh --dry-run --limit 16

# Run 32 matched pairs end to end.
./scripts/run_experiments.sh --limit 64
```

The full default uses 128 matched pairs (256 records). Hidden states are collected for every record. Expensive full attention tensors are requested only for the configured `attention_limit` subset; the default is 64 records.

Important outputs:

```text
summary.md
counterfactual_result.png
phase_probe_summary.csv
paired_direction_summary.csv
phase_transfer_summary.csv
attention_asymmetry_summary.csv
text_baseline_summary.csv
output_control.csv
activations_by_phase.npz
records_with_responses.jsonl
```

`pre_selector` is the built-in negative control. If it is substantially predictive, inspect the paired dataset/phase alignment before interpreting later phases.

To use another model or run name:

```bash
MODEL_NAME=Qwen/Qwen3-14B RUN_NAME=counterfactual-qwen3-14b ./scripts/run_experiments.sh
```

To disable the attention follow-up while retaining all hidden-state analyses, set `collect_attentions: false` in a copied config and run:

```bash
CONFIG=configs/my_config.yaml ./scripts/run_experiments.sh
```
