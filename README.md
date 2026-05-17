# Latent Intent

**Can an LLM represent which private goal is active when prompt vocabulary and observed output are held constant?**

This project probes a synthetic, prompt-injected private objective under matched counterfactual conditions. Every pair contains the same two objectives, the same public task, and the same teacher-forced response. The only difference is whether objective **A** or **B** is marked active.

| | Counterfactual 1 | Counterfactual 2 |
|---|---|---|
| Objective A | same text | same text |
| Objective B | same text | same text |
| Active selector | **A** | **B** |
| Public response | `Status: review complete.` | `Status: review complete.` |

Which slot contains the harmful objective is randomized within every objective domain, so `A` versus `B` alone is not predictive. Probe evaluation also holds out entire objective domains such as phishing, audit tampering, credential access, and surveillance.

The key negative control is **pre-selector**: paired prompts are byte-identical up to that point, so the selected-goal label should be unreadable there. The experiment then asks whether a representation emerges after the selector and persists through an identical response.

## Result

In the first full `Qwen/Qwen3-8B` counterfactual run (128 matched pairs across 16 objective domains), the control behaved exactly as intended: **pre-selector AUROC was 0.500 at every layer and paired activation deltas were zero**.

After the selector, the selected objective became almost perfectly linearly decodable on held-out objective domains. Layers 16-33 reached mean AUROC 1.00 immediately after selection; during the identical teacher-forced response, layer 23 reached AUROC 1.00 and balanced accuracy 1.00 in all five held-out-domain folds. The paired geometry was also strongly shared across domains: final-prompt layer 28 had held-out cosine 0.743 with 128/128 pair directions positive, and response-mean layer 35 had held-out cosine 0.756 with 126/128 positive.

The narrow headline is:

> Across held-out objective domains, the selected goal is unreadable before the selector, becomes nearly perfectly linearly decodable after selection, and remains decodable while the model produces an identical response.

The current analysis adds two stricter controls before treating this as a final result: a `selected_objective_tfidf` baseline that explicitly parses the selected objective from the prompt, and a paired-direction null that sign-flips whole held-out objective domains rather than treating pairs from the same domain as independent.

## Run

```bash
git pull
./scripts/run_experiments.sh
```

The script reuses an active research environment when possible, installs only missing dependencies, downloads `Qwen/Qwen3-8B` if needed, and runs the full counterfactual experiment.

For a quick server check first:

```bash
./scripts/run_experiments.sh --limit 64
```

To inspect the paired dataset without loading a model:

```bash
./scripts/run_experiments.sh --dry-run --limit 16
```

The explicit full config is `configs/qwen3_8b_counterfactual.yaml`.

For an overnight robustness run over three independent dataset/slot-assignment seeds:

```bash
bash scripts/run_overnight.sh
```

Override the replication seeds with, for example:

```bash
LATENT_INTENT_SEEDS="20260607 20260818 20260819 20260820 20260821" \
  bash scripts/run_overnight.sh
```

Each replication gets its own result directory and log under `logs/overnight/`.

## Measurements

The run produces three complementary activation-level tests:

1. **Held-out-domain probes** — harmful-active vs benign-active linear probes at `pre_selector`, `post_selector`, public-task, and response phases, with cross-validation grouped by `scenario_id`.
2. **Paired activation directions** — for each matched pair, compute `h_harmful-active - h_benign-active`; learn a direction without one objective domain and evaluate it on that held-out domain. Statistical significance uses grouped `scenario_id` sign flips; with 16 domains, all 2^16 null assignments are enumerated exactly.
3. **Active-objective attention** — compare each head's attention density to the active objective against the simultaneously present inactive objective.

Lexical controls include TF-IDF on the output, full prompt, selector, objective pair, and the **parsed selected objective itself**. The last of these is deliberately stronger than whole-prompt bag-of-words because it can express the selector/objective interaction. The fixed-output match rate is checked directly.

A completed run writes one main figure:

```text
results/<run>/counterfactual_result.png
```

It combines phase-by-layer held-out-domain AUROC with the held-out paired-direction geometry. `phase_probe_summary.csv` reports both the across-layer mean and the best layer for each phase, so strong localized representations are not hidden by averaging over early layers.

## Interpretation

This does **not** test autonomous malicious intent or prove a deception circuit. The objective is synthetic and explicitly supplied in the prompt. The narrow question is whether selecting one of two simultaneously present goals induces an internal representation that can be tracked after the selection event while observable behavior is fixed.
