# Latent Intent Probes

## A Private Objective Does Not Stay In One Place

We tested a narrow question about latent intent: if a model is given a private
objective and asked to keep the surface answer mundane, where is that objective
linearly readable, and does the same representation persist through the forward
pass?

The private objective here is synthetic and prompt-injected. This is not evidence
of autonomous deception. The point is more surgical: many deception probes assume
there is a convenient hidden-state direction for "the intent." We instead ask
whether the signal is phase-local, whether it transfers across computational
phases, and whether attention to the objective span gives a concrete handle.

Each example contains a benign or harmful private objective plus a public cover
task. We record activations at six phases:

`objective_span`, `post_objective`, `public_task_span`, `final_prompt`,
`response_first`, and `response_mean`.

We then fit linear harmful-vs-benign probes by phase/layer, train probes on one
phase and test them on another, compare against TF-IDF text baselines, check
output leakage, and, for Qwen3, rank attention heads by goal-anchor asymmetry:

```text
attention_density_to_objective - attention_density_to_public_task
```

## Main Result

On the full `Qwen/Qwen2.5-1.5B-Instruct` run, the private-objective label is
almost perfectly readable at the objective text and immediately afterward:

| phase | AUROC |
|---|---:|
| objective_span | 1.000 |
| post_objective | 0.999 |
| public_task_span | 0.883 |
| response_mean | 0.789 |
| final_prompt | 0.782 |
| response_first | 0.726 |

The nontrivial part is not that the objective text is readable. It is that the
signal survives after the prompt has moved on, especially in mid layers. For
Qwen2.5, the best later-phase layers cluster around layer 16:

| phase | best layer | AUROC |
|---|---:|---:|
| public_task_span | 16 | 0.965 |
| response_mean | 16 | 0.916 |
| final_prompt | 14 | 0.878 |
| response_first | 16 | 0.845 |

This is already a useful warning for probe work: the public-task and response
states carry label information even when the output text itself is much weaker.
The output TF-IDF baseline reaches only AUROC `0.662`, while activation probes
remain substantially stronger.

## The Crispy Bit: Phase-Transfer Asymmetry

The strongest pattern is an asymmetry in transfer. Early objective-phase probes
transfer poorly to later response states:

| transfer | AUROC |
|---|---:|
| objective_span -> response_first | 0.571 |
| post_objective -> response_first | 0.544 |
| objective_span -> public_task_span | 0.615 |

But later probes often transfer backward to the objective/post-objective phases:

| transfer | AUROC |
|---|---:|
| response_mean -> objective_span | 0.961 |
| response_mean -> post_objective | 0.893 |
| response_first -> objective_span | 0.888 |

That suggests the private-objective signal is not simply copied forward as one
stable linear direction. A better hypothesis is that the model re-encodes the
objective into a phase-specific state as it moves from "reading the private
objective" to "performing the public task" to "generating the answer."

## Qwen3-8B Replication And Attention Handle

We then ran a fast `Qwen/Qwen3-8B` replication with `64` balanced binary examples,
attention collection enabled, and thinking disabled. Because this run is small,
treat it as corroborating evidence, not a final estimate.

The early readout replicated:

| phase | AUROC |
|---|---:|
| objective_span | 0.985 |
| post_objective | 0.984 |
| final_prompt | 0.758 |
| public_task_span | 0.727 |
| response_mean | 0.670 |
| response_first | 0.433 |

Layer-specific late-phase signal was still present: `response_mean` layers
18-20 reached about AUROC `0.895-0.899`.

The transfer matrix again separates early-objective space from later task/answer
space. Early objective phases transfer to each other almost perfectly:

| transfer | AUROC |
|---|---:|
| objective_span -> post_objective | 0.999 |
| post_objective -> objective_span | 0.999 |

But early-to-late transfer is much weaker:

| transfer | AUROC |
|---|---:|
| objective_span -> public_task_span | 0.570 |
| objective_span -> response_first | 0.587 |

Meanwhile later states transfer strongly among themselves and into `final_prompt`:

| transfer | AUROC |
|---|---:|
| public_task_span -> final_prompt | 0.971 |
| response_first -> final_prompt | 0.958 |
| response_mean -> final_prompt | 0.927 |
| response_first -> public_task_span | 0.910 |

Attention collection also produced a concrete candidate phenomenon. Several heads
showed signed harmful-vs-benign differences in objective-vs-task attention
density. The largest effects were mostly benign-positive, meaning benign examples
had higher goal-anchor asymmetry than harmful examples:

| query phase | layer | head | Cohen's d | AUROC from raw asymmetry |
|---|---:|---:|---:|---:|
| objective_span | 22 | 20 | -1.59 | 0.129 |
| objective_span | 30 | 13 | -1.39 | 0.159 |
| response_mean | 24 | 23 | -1.36 | 0.151 |

The negative AUROCs are informative, not failures: sign-inverting the feature
would classify the label. There were harmful-positive heads too, but weaker:

| query phase | layer | head | Cohen's d | AUROC |
|---|---:|---:|---:|---:|
| post_objective | 10 | 17 | 1.04 | 0.795 |
| post_objective | 8 | 12 | 0.97 | 0.747 |
| response_mean | 18 | 10 | 0.89 | 0.761 |

This gives a specific follow-up target: not "find deception neurons," but inspect
the heads whose objective-vs-task attention density changes sign by label, and
ask how they interact with the mid-layer phase-transfer direction.

## Caveats

The Qwen2.5 full run has strong prompt/objective text baselines:

| baseline | AUROC |
|---|---:|
| objective_text_tfidf | 0.997 |
| prompt_tfidf | 0.991 |
| output_tfidf | 0.662 |

So the next dataset should lexically match harmful and benign objectives more
tightly. The Qwen3 limited run was better on this front:

| baseline | AUROC |
|---|---:|
| objective_text_tfidf | 0.842 |
| prompt_tfidf | 0.778 |
| output_tfidf | 0.341 |

Output leakage exists and should be controlled. In the Qwen2.5 full run, `6.9%`
of responses had harmful/private lexicon hits and `21.7%` mentioned private,
hidden, internal, harness, or experiment language. In the Qwen3 limited run, any
leakage was `28.1%` for both benign and harmful labels.

## Takeaway

The interesting object is not a single latent-intent direction. It is a temporal
pattern: early objective readout, partial survival into public-task and response
states, asymmetric phase transfer, and signed attention heads that route
differently to objective versus task spans. That is a more precise next target
than generic "deception probing."

## Running

Install and run on the inference server:

```bash
cd latent-intent-probes
cp .env.example .env
$EDITOR .env
./scripts/run_experiments.sh
```

The token parser accepts `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HF_HUB_TOKEN`,
lowercase `hf_token`, and lowercase `huggingface_token`.

Run the Qwen3 attention config:

```bash
CONFIG=configs/qwen3_8b_attention.yaml ./scripts/run_experiments.sh
```

Fast smoke test:

```bash
CONFIG=configs/qwen3_8b_attention.yaml ./scripts/run_experiments.sh --limit 64
```

Important outputs:

```text
summary.md
phase_probe_summary.csv
activation_probe_summary.csv
phase_transfer_summary.csv
attention_asymmetry_summary.csv
records_with_responses.jsonl
activations_by_phase.npz
attention_asymmetry_rows.csv
phase_layer_auroc.png
phase_transfer_auroc.png
```
