## What were you trying to figure out, and why did it matter?

I spent more than 20 hours on a small AI safety research project about whether
private-objective signals in language models are better understood as a static
probe target or as a temporally changing computation. The motivating question
was: if a model is prompted with a hidden benign or harmful private objective
while being asked to produce an ordinary public answer, where in the forward pass
is that private objective linearly readable, and does the same representation
persist into the response? This matters because a lot of latent-deception or
intent probing work implicitly treats "the objective" as a thing one can probe
at a convenient token. I wanted to test whether that assumption breaks once we
look across phases of the computation. The repo is
`latent-intent-probes`; the resulting short writeup is in `README.md`.

## What did you actually do?

I built a small experiment pipeline around Qwen instruction models. The dataset
paired harmful and benign private objectives with mundane cover tasks, plus
neutral decoys. The runner generated responses, then did a teacher-forced pass
over prompt plus response and extracted activations at six phases:
`objective_span`, `post_objective`, `public_task_span`, `final_prompt`,
`response_first`, and `response_mean`. I fit linear probes by phase and layer,
phase-transfer probes that train on one phase and test on another, TF-IDF text
baselines, output leakage checks, and attention-asymmetry summaries measuring
`attention_density_to_objective - attention_density_to_public_task`. Outputs
included code, configs, run artifacts, plots, and a short blog-style README. I
ran a full Qwen2.5-1.5B experiment and a smaller Qwen3-8B attention run on a
remote GPU server.

## What surprised you?

The most interesting result was not that the objective text was readable; that
was expected. The surprise was the phase-transfer asymmetry. Early objective
probes transferred poorly to later response phases, but later response-mean
probes often transferred back to the objective and post-objective phases. That
suggests the signal is not simply copied forward as one stable linear direction.
It looks more like the model re-encodes the objective into a phase-specific
state as it moves from reading the instruction to performing the public task.
The Qwen3 run also produced concrete attention-head candidates with signed
harmful-vs-benign differences.

## What did you do wrong, or what was suboptimal?

The first dataset was too lexically separable. In the full Qwen2.5 run, prompt
and objective TF-IDF baselines were very strong, so some activation signal could
reflect obvious lexical differences rather than a more abstract objective
representation. I also initially ran with attention collection disabled, which
meant the first run could not test the head-level hypothesis. Operationally, I
lost time debugging remote GPU setup and dependency drift because the run script
originally reinstalled dependencies too eagerly.

## What's still unresolved or unknown?

The main unresolved question is whether the phase-transfer asymmetry survives
stronger controls: lexically matched benign/harmful objectives, more templates,
more models, and larger sample sizes. The attention-head result is also only a
hypothesis generator; it needs causal interventions or activation/attention
patching before I would claim a mechanism. I also do not yet know whether this
pattern says anything about naturally arising deceptive behavior, since the
private objective here is explicitly prompt-injected.
