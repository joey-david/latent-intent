# Blogpost Outline

## Working Title

**When Does a Selected Goal Appear in an LLM's Hidden State?**

## Thesis

A probe that distinguishes harmful from benign objective text can succeed for an uninteresting reason: the two prompts contain different words. The cleaner question is whether an internal state tracks **which of two simultaneously present objectives has been selected**.

## Counterfactual Design

Each matched pair contains the same:

- objective A;
- objective B;
- public task;
- teacher-forced response.

Only this line changes:

```text
ACTIVE OBJECTIVE: A
```

versus:

```text
ACTIVE OBJECTIVE: B
```

Which slot contains the harmful objective is balanced within objective domains. The response is always `Status: review complete.`

This creates a causal timeline:

```text
read A + B -> pre-selector -> selector -> post-selector -> public task -> identical response
```

`pre-selector` is a hard negative control because the two conditions are identical up to that token.

## Measurements

1. Phase-by-layer linear probes with complete objective domains held out.
2. Phase-transfer probes to test whether the selected-goal representation changes form.
3. Paired activation differences `h_harmful-active - h_benign-active`, evaluated as a shared direction on held-out domains with a sign-flip null.
4. Per-head attention density to the active objective minus the inactive objective.
5. Prompt, selector, objective-pair, and output TF-IDF controls.

## Main Figure

Two stacked heatmaps:

- held-out-domain selected-goal probe AUROC by phase/layer;
- held-out cosine of the shared paired counterfactual direction by phase/layer.

The visually clean result would be chance/zero before the selector followed by emergence after the selector and persistence, or systematic transformation, into the identical response.

## Claims Worth Making If Supported

- Selecting one of two simultaneously present objectives creates a decodable hidden-state difference that generalizes across objective domains.
- The representation emerges only after the selector, validating the temporal control.
- A shared counterfactual direction exists across semantically different objectives.
- Some attention heads preferentially route computation through the selected objective after selection.

## Claims To Avoid

- This is not autonomous malicious intent: the objectives are synthetic and prompt-injected.
- Probe accuracy alone is not a deception detector.
- Active-objective attention is not automatically a causal circuit.
- Results from Qwen3-8B do not establish universality across models.

## Follow-Ups

- Activation patch the selected-goal direction across counterfactual pairs.
- Ablate heads with strong active-vs-inactive objective preference.
- Replicate on base vs instruct models and other model families.
- Replace harmful/benign semantics with non-safety goal pairs to test whether the direction is goal-selection-general or valence-specific.
