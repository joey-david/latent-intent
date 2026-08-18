# Counterfactual Latent Intent: counterfactual-latent-intent-qwen3-8b

Two prompt conditions contain the same objective pair, public task, and teacher-forced response. They differ only in whether objective A or B is marked active. Which slot contains the harmful objective is randomized across pairs.

## Controls

- Model: `Qwen/Qwen3-8B`
- Matched counterfactual pairs: `32`
- Cross-validation group: held-out `scenario_id` objective domains
- Exact fixed-output match rate: `1.000`
- `pre_selector` is a negative control: pair members are byte-identical up to the A/B selector.

## Probe AUROC By Phase

| phase            |   auroc_mean |   auroc_std |   bal_acc_mean |
|:-----------------|-------------:|------------:|---------------:|
| final_prompt     |     0.753868 |    0.242326 |       0.727431 |
| post_selector    |     0.843229 |    0.240583 |       0.816435 |
| pre_selector     |     0.5      |    0        |       0.5      |
| public_task_span |     0.766069 |    0.223393 |       0.687847 |
| response_first   |     0.763281 |    0.236786 |       0.723032 |
| response_mean    |     0.778405 |    0.239319 |       0.737269 |

## Best Phase-Layer Readouts

| phase         |   layer |   auroc_mean |   auroc_std |   bal_acc_mean |
|:--------------|--------:|-------------:|------------:|---------------:|
| final_prompt  |      20 |            1 |           0 |       0.983333 |
| post_selector |      29 |            1 |           0 |       0.983333 |
| post_selector |      30 |            1 |           0 |       0.983333 |
| post_selector |      31 |            1 |           0 |       0.983333 |
| post_selector |      16 |            1 |           0 |       1        |
| post_selector |      17 |            1 |           0 |       1        |
| post_selector |      18 |            1 |           0 |       1        |
| post_selector |      32 |            1 |           0 |       0.983333 |

## Lexical Controls

| model_kind           |   auroc_mean |   auroc_std |   bal_acc_mean |
|:---------------------|-------------:|------------:|---------------:|
| objective_pair_tfidf |     0.5      |    0        |       0.5      |
| output_tfidf         |     0.5      |    0        |       0.5      |
| prompt_tfidf         |     0.613194 |    0.177519 |       0.558333 |
| selector_tfidf       |     0.558333 |    0.14313  |       0.558333 |

## Shared Counterfactual Directions

| phase         |   layer |   n_pairs |   delta_norm_mean |   direction_coherence |   heldout_cosine_mean |   heldout_positive_rate |   permutation_p |
|:--------------|--------:|----------:|------------------:|----------------------:|----------------------:|------------------------:|----------------:|
| final_prompt  |      25 |        32 |           18.5829 |              0.714437 |              0.683586 |                       1 |      0.00497512 |
| final_prompt  |      27 |        32 |           24.6111 |              0.715199 |              0.683036 |                       1 |      0.00497512 |
| final_prompt  |      26 |        32 |           21.816  |              0.714587 |              0.682951 |                       1 |      0.00497512 |
| final_prompt  |      28 |        32 |           28.3023 |              0.713742 |              0.681005 |                       1 |      0.00497512 |
| final_prompt  |      22 |        32 |           10.5843 |              0.709332 |              0.679432 |                       1 |      0.00497512 |
| final_prompt  |      29 |        32 |           31.388  |              0.712482 |              0.67932  |                       1 |      0.00497512 |
| response_mean |      35 |        32 |            9.8255 |              0.709508 |              0.678237 |                       1 |      0.00497512 |
| final_prompt  |      30 |        32 |           35.142  |              0.710741 |              0.677494 |                       1 |      0.00497512 |

`heldout_cosine_mean` evaluates a mean harmful-minus-benign direction on objective domains excluded from that direction's construction. `permutation_p` uses random sign flips of paired deltas.

## Phase Transfer

| train_phase      | test_phase       |   auroc_mean |   auroc_std |   bal_acc_mean |
|:-----------------|:-----------------|-------------:|------------:|---------------:|
| response_mean    | post_selector    |     1        |  0          |       0.832639 |
| post_selector    | post_selector    |     1        |  0          |       1        |
| response_mean    | response_mean    |     0.999074 |  0.00507151 |       0.963889 |
| final_prompt     | final_prompt     |     0.993924 |  0.0113958  |       0.953472 |
| response_first   | response_first   |     0.98588  |  0.0220723  |       0.925694 |
| response_first   | response_mean    |     0.979514 |  0.0503594  |       0.729167 |
| public_task_span | public_task_span |     0.978646 |  0.0403671  |       0.863889 |
| final_prompt     | post_selector    |     0.976331 |  0.0916268  |       0.64375  |
| response_mean    | response_first   |     0.971644 |  0.0433712  |       0.620139 |
| final_prompt     | response_mean    |     0.967824 |  0.0446521  |       0.551389 |
| public_task_span | response_mean    |     0.967419 |  0.0392034  |       0.588194 |
| response_mean    | final_prompt     |     0.96684  |  0.0751687  |       0.625694 |
| response_first   | final_prompt     |     0.948206 |  0.0923989  |       0.6625   |
| final_prompt     | response_first   |     0.93287  |  0.0761522  |       0.541667 |
| post_selector    | response_mean    |     0.925579 |  0.0867286  |       0.513194 |
| public_task_span | final_prompt     |     0.919213 |  0.0752328  |       0.565972 |
| public_task_span | post_selector    |     0.90191  |  0.139853   |       0.7625   |
| public_task_span | response_first   |     0.880266 |  0.103218   |       0.516667 |
| response_first   | post_selector    |     0.833391 |  0.134548   |       0.58125  |
| response_mean    | public_task_span |     0.827662 |  0.111072   |       0.511111 |

## Active-vs-Inactive Objective Attention

| query_phase   |   layer |   head |   mean_active_minus_inactive |   cohens_d_vs_zero |   abs_cohens_d |   active_preference_rate |   harmful_mean |   benign_mean |   n |
|:--------------|--------:|-------:|-----------------------------:|-------------------:|---------------:|-------------------------:|---------------:|--------------:|----:|
| post_selector |       9 |     11 |                   0.00759028 |            2.44945 |        2.44945 |                 1        |     0.00783432 |    0.00734625 |  64 |
| post_selector |      14 |     11 |                   0.0108276  |            1.94666 |        1.94666 |                 0.984375 |     0.00966211 |    0.0119931  |  64 |
| post_selector |      18 |     13 |                  -0.0106332  |           -1.91618 |        1.91618 |                 0        |    -0.00947172 |   -0.0117946  |  64 |
| post_selector |      19 |     30 |                  -0.0272078  |           -1.89439 |        1.89439 |                 0        |    -0.0324564  |   -0.0219593  |  64 |
| post_selector |      18 |     12 |                  -0.0110105  |           -1.87625 |        1.87625 |                 0        |    -0.0120325  |   -0.00998849 |  64 |
| post_selector |      13 |     14 |                   0.0193416  |            1.72708 |        1.72708 |                 1        |     0.0193725  |    0.0193108  |  64 |
| post_selector |      24 |     21 |                   0.00310687 |            1.6528  |        1.6528  |                 0.96875  |     0.00270304 |    0.00351071 |  64 |
| post_selector |      21 |     31 |                   0.0107099  |            1.65133 |        1.65133 |                 0.984375 |     0.00986914 |    0.0115507  |  64 |
| post_selector |      16 |     23 |                   0.0465421  |            1.57266 |        1.57266 |                 0.921875 |     0.0382024  |    0.0548818  |  64 |
| post_selector |      20 |      9 |                   0.00204875 |            1.56259 |        1.56259 |                 0.90625  |     0.00184976 |    0.00224774 |  64 |
| post_selector |      20 |     15 |                   0.0341613  |            1.49808 |        1.49808 |                 0.921875 |     0.0287424  |    0.0395803  |  64 |
| post_selector |      18 |     15 |                   0.0310609  |            1.48934 |        1.48934 |                 1        |     0.0235163  |    0.0386056  |  64 |
| post_selector |      15 |      5 |                   0.00385283 |            1.45882 |        1.45882 |                 0.96875  |     0.00358196 |    0.0041237  |  64 |
| post_selector |      17 |     15 |                   0.0432182  |            1.44283 |        1.44283 |                 1        |     0.034691   |    0.0517453  |  64 |
| post_selector |      19 |      0 |                  -0.00991178 |           -1.42749 |        1.42749 |                 0.03125  |    -0.01029    |   -0.00953361 |  64 |
| post_selector |      21 |     18 |                   0.0115025  |            1.40746 |        1.40746 |                 0.984375 |     0.0136887  |    0.00931633 |  64 |
| post_selector |      16 |     17 |                  -0.0218876  |           -1.38643 |        1.38643 |                 0        |    -0.0239444  |   -0.0198307  |  64 |
| post_selector |       9 |      8 |                   0.00206215 |            1.35551 |        1.35551 |                 0.9375   |     0.0021504  |    0.00197391 |  64 |
| post_selector |      17 |      4 |                   0.0225241  |            1.34813 |        1.34813 |                 1        |     0.0149593  |    0.0300889  |  64 |
| post_selector |      20 |     12 |                  -0.00477084 |           -1.34227 |        1.34227 |                 0.015625 |    -0.00536565 |   -0.00417603 |  64 |

## Main Artifacts

- `counterfactual_result.png`: probe AUROC and paired-direction heatmaps
- `phase_probe_summary.csv`: phase-level activation probe results
- `paired_direction_summary.csv`: counterfactual direction statistics
- `phase_transfer_summary.csv`: train-phase to test-phase generalization
- `attention_asymmetry_summary.csv`: active-vs-inactive objective attention heads
- `text_baseline_summary.csv`: lexical negative controls
- `output_control.csv`: verifies the public response is identical

## Interpretation

This experiment studies a synthetic, prompt-injected selected objective, not autonomous malicious intent. The interesting evidence is whether a label that is undefined before the selector becomes readable afterward, generalizes to held-out objective domains, persists into an identical response, and yields a shared paired activation direction or active-objective attention preference.
