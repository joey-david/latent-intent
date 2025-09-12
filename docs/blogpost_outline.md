# Blogpost Outline

## Working Title

When Does Latent Mischief Exist? Phase Probes And Goal-Anchor Attention In LLMs

## Thesis

Current deception-probe work often treats deceptive or suppressed-objective state
as something you can read from a convenient token position. This post tests a
more mechanistic assumption: the signal may be temporally localized. It may
appear while the model reads the private objective, persist into answer drafting,
or disappear before token emission.

## Phenomenon

Goal-anchor attention asymmetry:

```text
attention_density_to_private_objective_span - attention_density_to_public_task_span
```

The hypothesis is not that this is "the deception circuit." The hypothesis is
that a phase-local attention pattern may tell us whether the model is still
routing computation through the private objective after the answer has moved to a
mundane public task.

## Experiment

Use paired prompts that differ in a hidden benign-vs-harmful objective but share
the same public cover task. The model is instructed to emit only a normal status
note, not to disclose the objective. Record activations and attentions over:

- private-objective span;
- post-objective token;
- public-task span;
- final prompt token;
- first generated token;
- mean generated response.

Then train phase-wise linear probes, phase-transfer probes, output/prompt/objective
text baselines, and rank heads by goal-anchor attention asymmetry.

## Expected Figures

- Phase-by-layer activation AUROC heatmap.
- Phase-transfer AUROC heatmap.
- Top heads by harmful-vs-benign goal-anchor asymmetry.
- Output leakage rate.
- Example rows where output looks neutral but response-phase activations remain
  predictive.

## Claims Worth Making If Results Support Them

- The private-objective signal is not necessarily a static thing; it can be
  phase-local.
- If objective-phase probes do not transfer to response phases, the standard
  "probe one token" setup is underspecified.
- If response-phase probes remain strong while output leakage is low, behavioral
  monitoring is missing a hidden-state residue.
- If a small set of heads shows strong goal-anchor asymmetry, follow-up work can
  ablate or patch those heads.

## Claims To Avoid

- Do not claim the model has autonomous malicious intent.
- Do not claim goal-anchor attention is a universal deception circuit.
- Do not claim safety training failed; this repo does not fine-tune or evaluate
  a sleeper-agent training protocol.
- Do not infer deployment risk from one synthetic prompt family.

## Follow-Up Experiments

- Head ablation or activation patching on the top asymmetry heads.
- Cross-model comparison across Qwen, Gemma, Llama, and Mistral variants.
- Prompt variants where the private objective is paraphrased or moved earlier.
- A redacted lexical control where harmful and benign objectives share most words.
- Compare base vs instruct variants to test whether alignment moves the phase
  where the signal is readable.
