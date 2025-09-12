# Blogpost Outline

## Working Title

Latent Mischief Without Surface Mischief: Probing Suppressed Objective Signals in
LLM Hidden States

## Thesis

Behavioral monitoring can miss a private objective if the final answer is forced
to look mundane. A small activation-probing experiment can test whether the
private objective remains linearly decodable from internal states even when the
generated text stays neutral.

## Experiment

Use paired prompts that differ in a hidden benign-vs-harmful objective but share
the same public cover task. The model is instructed to emit only a normal status
note, not to disclose the objective. Record final-prompt-token activations across
layers, train layer-wise probes, and compare against output-only TF-IDF baselines.

## Expected Figures

- Layer-wise activation AUROC.
- Output leakage rate.
- Activation probe score versus output-text baseline.
- A small table of generated examples where the output looks neutral but the
  activation probe is confident.

## Claims Worth Making If Results Support Them

- The generated response may be a poor proxy for the model state induced by a
  private objective.
- Mid-layer activations may preserve the private-objective distinction more than
  late layers or output text.
- Output-only monitors can look clean while activation probes detect the
  experimental condition.

## Claims To Avoid

- Do not claim the model has autonomous malicious intent.
- Do not claim the probe has found a universal deception direction.
- Do not claim safety training failed; this repo does not fine-tune or run a
  sleeper-agent training protocol.
- Do not infer real-world deployment risk from a single synthetic prompt family.

## Follow-Up Experiments

- Cross-model comparison across Qwen, Gemma, Llama, and Mistral variants.
- Probe transfer from one prompt-template family to another.
- Activation steering along the learned direction to see whether output leakage
  increases or decreases.
- A redacted-prompt control where surface words are normalized more aggressively.
