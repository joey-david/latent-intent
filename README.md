# Latent Intent

**Counterfactual goal binding in language-model hidden states.**

Can we tell which of two simultaneously present goals a model has selected when the goal texts and observable continuation are held fixed?

![Held-out-domain decoding and shared counterfactual direction](results/20260818T003055Z-counterfactual-latent-intent-qwen3-8b-seed-20260818/counterfactual_result.png)

## Result

On `Qwen/Qwen3-8B`, across **3 independent seeds**, **128 matched pairs**, and **16 objective domains**:

- Before the selector: **AUROC = 0.500 at every layer** in every run.
- After selection: best held-out-domain probe AUROC reaches **1.000**, including during the identical response.

I have yet to show causality, but experiments are ongoing!

## Run

```bash
git clone https://github.com/joey-david/latent-intent.git
cd latent-intent
./scripts/run_experiments.sh
```

Three-seed replication:

```bash
bash scripts/run_overnight.sh
```

Results are written to `results/<run>/`.

## Scope

This is a controlled representation study, not evidence of autonomous malicious intent or a deception circuit (yet :)). The selected objective is explicitly supplied in the prompt; the experiment isolates the internal effect of selecting it.
