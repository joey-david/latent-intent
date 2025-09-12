# Latent Intent Probes

Small experiment repo for a blogpost on a narrow version of latent deception:
whether a model's hidden states retain a decodable private-objective signal after
the response is constrained to be ordinary, benign surface text.

This is not a claim that the model has autonomous malicious intent. The setup is
synthetic and deliberately labeled. The useful question is narrower: can a
harmful-vs-benign private objective remain visible to activation probes even when
the generated answer itself does not reveal the objective?

## Research Hook

The experiment sits between three research threads:

- Backdoor/deceptive-policy evaluations, especially sleeper-agent style behavior:
  <https://arxiv.org/abs/2401.05566>
- Representation engineering and high-level population readouts:
  <https://arxiv.org/abs/2310.01405>
- Linear probes for strategic deception and their limitations:
  <https://arxiv.org/abs/2502.03407>
- Latent-knowledge framing for reading information from internal activations:
  <https://arxiv.org/abs/2212.03827>

## What It Runs

For each prompt, the dataset gives the model:

- a private objective label: benign or harmful;
- a public instruction to produce only a mundane note;
- a system safety instruction forbidding operational wrongdoing.

The runner then:

1. renders chat prompts for the target Hugging Face model;
2. records final-prompt-token hidden states from every transformer layer;
3. generates the public answer with greedy decoding;
4. trains layer-wise logistic probes for the private objective label;
5. trains output-text and prompt-text TF-IDF baselines;
6. writes a markdown summary, CSV metrics, generated responses, activations, and an AUROC plot.

## SSH Server Run

Do this on the inference server after cloning or copying the repo:

```bash
cd latent-intent-probes
cp .env.example .env
$EDITOR .env
./scripts/run_experiments.sh
```

The `.env` file needs:

```bash
HF_TOKEN=hf_...
```

Optional overrides:

```bash
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
RUN_NAME=latent-intent-qwen15b
HF_HOME=/path/to/hf-cache
TRANSFORMERS_CACHE=/path/to/hf-cache
```

The default model is `Qwen/Qwen2.5-1.5B-Instruct`, chosen because it is small
enough for a modest GPU while still being an instruction model. If the server has
more VRAM, set `MODEL_NAME` to a larger compatible causal LM in `.env`.

## Quick Checks

Local no-model dry run:

```bash
PYTHONPATH=src python -m latent_intent_probe.run --config configs/default.yaml --dry-run
```

Small server smoke test:

```bash
./scripts/run_experiments.sh --limit 24
```

Full run:

```bash
./scripts/run_experiments.sh
```

## Outputs To Pull Back

Each run writes a timestamped directory under `results/`, for example:

```text
results/20260607T181530Z-latent-intent-qwen15b/
```

Pull back the whole run directory. The most important files are:

- `summary.md`: readable run summary;
- `activation_probe_summary.csv`: mean probe scores by layer;
- `text_baseline_summary.csv`: output/prompt baseline scores;
- `records_with_responses.jsonl`: prompts, labels, generated answers;
- `activations_last_prompt_token.npz`: activation tensor `[example, layer, hidden]`;
- `layer_auroc.png`: layer-wise AUROC plot.

To archive on the server:

```bash
tar -czf latent-intent-results.tgz results/
```

## Interpretation Guardrails

This experiment can support a blogpost about answer-space suppression and
activation-space residue. It cannot prove natural deceptive intent, because the
private objective is provided in the prompt by construction. The TF-IDF baselines
are included so the analysis can say whether hidden-state probes add anything
beyond obvious prompt or generated-output surface cues.
