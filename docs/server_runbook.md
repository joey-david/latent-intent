# Server Runbook

The default experiment is ready to run with one script invocation.

```bash
cd latent-intent
git pull
./scripts/run_experiments.sh
```

The runner reuses an active/existing environment when possible, installs only missing dependencies, loads `Qwen/Qwen3-8B`, and writes a timestamped run under `results/`.

Before the full run, two useful checks are:

```bash
# Build and inspect 8 matched pairs without loading the model.
./scripts/run_experiments.sh --dry-run --limit 16

# Run 32 matched pairs end to end.
./scripts/run_experiments.sh --limit 64
```

The full default uses 128 matched pairs (256 records). Hidden states are collected for every record. Expensive full attention tensors are requested only for the configured `attention_limit` subset; the default is 64 records.

The adjusted analysis includes:

- a parsed `selected_objective_tfidf` lexical baseline;
- a `phase_probe_summary.csv` that reports both all-layer means and best-layer performance;
- an exact domain-grouped paired-direction sign-flip null for the 16-domain design;
- corrected figure labels describing harmful-active vs benign-active decoding.

For an overnight robustness suite over three independent dataset/slot-assignment seeds:

```bash
bash scripts/run_overnight.sh
```

The default seeds are `20260607`, `20260818`, and `20260819`. Override them with:

```bash
LATENT_INTENT_SEEDS="20260607 20260818 20260819 20260820 20260821" \
  bash scripts/run_overnight.sh
```

Each replication writes its normal timestamped directory under `results/` and a terminal log under `logs/overnight/`. `RUN_SEED` can also be supplied directly for one replication:

```bash
RUN_SEED=20260818 \
RUN_NAME=counterfactual-latent-intent-qwen3-8b-seed-20260818 \
./scripts/run_experiments.sh
```

Important outputs:

```text
summary.md
counterfactual_result.png
phase_probe_summary.csv
activation_probe_summary.csv
paired_direction_summary.csv
phase_transfer_summary.csv
attention_asymmetry_summary.csv
text_baseline_summary.csv
output_control.csv
activations_by_phase.npz
records_with_responses.jsonl
```

`pre_selector` is the built-in hard negative control. If it is substantially predictive, inspect the paired dataset/phase alignment before interpreting later phases.

To use another model or run name:

```bash
MODEL_NAME=Qwen/Qwen3-14B RUN_NAME=counterfactual-qwen3-14b ./scripts/run_experiments.sh
```

To disable the attention follow-up while retaining all hidden-state analyses, set `collect_attentions: false` in a copied config and run:

```bash
CONFIG=configs/my_config.yaml ./scripts/run_experiments.sh
```
