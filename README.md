# Latent Intent Probes

Small experiment repo for a blogpost on a current, sharper question in latent
deception work:

> When does a private-objective signal exist in the forward pass, and does it
> line up with a concrete computation pattern?

The phenomenon under test is **goal-anchor attention asymmetry**: whether certain
layers/heads route more attention from key phases of the computation back to the
private-objective span than to the public-task span, and whether that asymmetry
differs for harmful vs benign private objectives.

This is not a claim that the model has autonomous malicious intent. The private
objective is synthetic and prompt-injected. The point is to test a SOTA
assumption: that "deception" or suppressed-objective signals can be treated as a
static probe target at a convenient token. This repo instead asks where the signal
appears, whether it transfers across phases, and whether an attention pattern
gives us a hypothesis for further mechanistic work.

## Research Hook

The experiment sits between these research threads:

- Backdoor/deceptive-policy evaluations, especially sleeper-agent style behavior:
  <https://arxiv.org/abs/2401.05566>
- Representation engineering and high-level population readouts:
  <https://arxiv.org/abs/2310.01405>
- Linear probes for strategic deception and their limitations:
  <https://arxiv.org/abs/2502.03407>
- Latent-knowledge framing for reading information from internal activations:
  <https://arxiv.org/abs/2212.03827>

## Battery Of Tests

For each prompt, the dataset gives the model:

- a private objective label: benign or harmful;
- a public cover task;
- a system safety instruction forbidding operational wrongdoing.

The runner records generated text and then does a teacher-forced forward pass
over prompt plus response. It extracts activations at six phases:

- `objective_span`: mean over the private-objective text;
- `post_objective`: final token of the private objective;
- `public_task_span`: mean over the public cover task;
- `final_prompt`: final prompt token before the response;
- `response_first`: first generated-response token;
- `response_mean`: mean over generated-response tokens.

It then runs:

- phase-by-layer linear probes for harmful vs benign private objective;
- phase-transfer probes, training on one phase and testing on another;
- output, prompt, and objective-text TF-IDF baselines;
- output leakage checks;
- head-level goal-anchor attention asymmetry:
  `attention_density_to_objective - attention_density_to_public_task`.

Useful patterns to look for:

- signal appears during `objective_span` but vanishes by `response_mean`;
- signal persists into response tokens despite neutral output;
- probes transfer from objective phase to response phase, suggesting a stable
  representation;
- probes do not transfer, suggesting phase-local computation;
- a small set of heads has strong harmful-vs-benign goal-anchor asymmetry.

## SSH Server Run

Do this on the inference server after cloning or copying the repo:

```bash
cd latent-intent-probes
cp .env.example .env
$EDITOR .env
./scripts/run_experiments.sh
```

The preferred `.env` variable is:

```bash
HF_TOKEN=hf_...
```

The runner also accepts `HUGGING_FACE_HUB_TOKEN`, `HF_HUB_TOKEN`, lowercase
`hf_token`, and lowercase `huggingface_token`.

Optional overrides:

```bash
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
RUN_NAME=temporal-latent-intent-qwen15b
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

Attention tensors are memory-heavy, so the default batch size is `2`. If VRAM is
tight, lower `model.batch_size` in `configs/default.yaml`.

## GPU Sanity Checks

The default config has `model.require_cuda: true`, so the run fails fast instead
of silently running on CPU. To check the environment:

```bash
./scripts/check_accel.sh
```

To also load the model and print the Transformers device map:

```bash
./scripts/check_accel.sh --check-model-placement
```

Healthy output should show:

- `torch.cuda_available: true`;
- a non-empty CUDA device list;
- `model_device_map` entries on CUDA, not `cpu` or `disk`.

If PyTorch reports a CPU build or no CUDA devices, fix the Python environment
before launching the experiment. For CUDA 12.4 servers, a typical install is:

```bash
source .venv/bin/activate
python -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu124 torch
```

Only set `ALLOW_CPU=true` if you intentionally want a slow CPU run.

## Outputs To Pull Back

Each run writes a timestamped directory under `results/`, for example:

```text
results/20260607T181530Z-latent-intent-qwen15b/
```

Pull back the whole run directory. The most important files are:

- `summary.md`: readable run summary;
- `phase_probe_summary.csv`: mean probe scores by phase;
- `activation_probe_summary.csv`: mean probe scores by phase and layer;
- `phase_transfer_summary.csv`: train-phase to test-phase summary;
- `attention_asymmetry_summary.csv`: ranked heads by harmful-vs-benign asymmetry;
- `records_with_responses.jsonl`: prompts, phase token counts, generated answers;
- `activations_by_phase.npz`: activation tensor `[example, phase, layer, hidden]`;
- `attention_asymmetry_rows.csv`: per-example, per-layer, per-head attention features;
- `phase_layer_auroc.png`: phase-by-layer AUROC heatmap;
- `phase_transfer_auroc.png`: transfer heatmap.

To archive on the server:

```bash
tar -czf latent-intent-results.tgz results/
```

## Interpretation Guardrails

This experiment can support a blogpost about temporal localization and a concrete
attention asymmetry hypothesis. It cannot prove natural deceptive intent. The
private objective is provided in the prompt by construction, and prompt/objective
text baselines are included to measure how much of the signal is surface-level.
