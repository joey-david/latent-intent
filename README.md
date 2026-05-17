# Latent Intent

**Can an LLM represent which private goal is active when prompt vocabulary and observed output are held constant?**

This project probes a synthetic, prompt-injected private objective under matched counterfactual conditions. Every pair contains the same two objectives, the same public task, and the same teacher-forced response. The only difference is whether objective **A** or **B** is marked active.

| | Counterfactual 1 | Counterfactual 2 |
|---|---|---|
| Objective A | same text | same text |
| Objective B | same text | same text |
| Active selector | **A** | **B** |
| Public response | `Status: review complete.` | `Status: review complete.` |

Which slot contains the harmful objective is randomized across pairs, so `A` versus `B` alone is not predictive. Probe evaluation also holds out entire objective domains such as phishing, audit tampering, credential access, and surveillance.

The key negative control is **pre-selector**: paired prompts are identical up to that point, so the selected-goal label should be unreadable there. The experiment then asks whether a representation emerges after the selector and persists through the identical response.

## Run

```bash
git pull
./scripts/run_experiments.sh
```

The script creates/updates `.venv`, installs the package, downloads `Qwen/Qwen3-8B`, and runs the default counterfactual experiment.

For a quick server check first:

```bash
./scripts/run_experiments.sh --limit 64
```

To inspect the paired dataset without loading a model:

```bash
./scripts/run_experiments.sh --dry-run --limit 16
```

The explicit full config is `configs/qwen3_8b_counterfactual.yaml`.

## Measurements

The run produces three complementary tests:

1. **Held-out-domain probes** — harmful-active vs benign-active linear probes at `pre_selector`, `post_selector`, public-task, and response phases, with cross-validation grouped by `scenario_id`.
2. **Paired activation directions** — for each matched pair, compute `h_harmful-active - h_benign-active`; test whether those directions align across objective domains and against a random sign-flip null.
3. **Active-objective attention** — compare each head's attention density to the active objective against the simultaneously present inactive objective.

TF-IDF baselines on the full prompt, selector, objective pair, and output are included as lexical controls. The fixed-output match rate is also checked directly.

## Result

**Counterfactual rerun pending.** The earlier version of this project used different harmful and benign objective text and therefore had a strong lexical confound; its headline probe numbers are intentionally not carried forward.

A completed run writes one main figure:

```text
results/<run>/counterfactual_result.png
```

It combines the phase-by-layer held-out-domain probe AUROC with the shared paired-direction result. The same directory contains `summary.md` and the underlying CSVs.

The result is interesting if the pre-selector control stays at chance while post-selector/public-task/response states become predictive on held-out objective domains. A shared paired direction or active-vs-inactive attention preference would provide a stronger mechanistic follow-up signal.

## Interpretation

This does **not** test autonomous malicious intent or prove a deception circuit. The objective is synthetic and explicitly supplied in the prompt. The narrow question is whether selecting one of two simultaneously present goals induces an internal representation that can be tracked after the selection event while observable behavior is fixed.
