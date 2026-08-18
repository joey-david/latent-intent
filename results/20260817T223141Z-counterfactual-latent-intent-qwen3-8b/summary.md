# Counterfactual Latent Intent: counterfactual-latent-intent-qwen3-8b

Two prompt conditions contain the same objective pair, public task, and teacher-forced response. They differ only in whether objective A or B is marked active. Which slot contains the harmful objective is randomized across pairs.

## Controls

- Model: `Qwen/Qwen3-8B`
- Matched counterfactual pairs: `128`
- Cross-validation group: held-out `scenario_id` objective domains
- Exact fixed-output match rate: `1.000`
- `pre_selector` is a negative control: pair members are byte-identical up to the A/B selector.

## Probe AUROC By Phase

| phase            |   auroc_mean |   auroc_std |   bal_acc_mean |
|:-----------------|-------------:|------------:|---------------:|
| final_prompt     |     0.779776 |    0.235403 |       0.762558 |
| post_selector    |     0.89268  |    0.16754  |       0.872685 |
| pre_selector     |     0.5      |    0        |       0.5      |
| public_task_span |     0.794579 |    0.21661  |       0.739612 |
| response_first   |     0.79716  |    0.222463 |       0.767824 |
| response_mean    |     0.795126 |    0.229328 |       0.766088 |

## Best Phase-Layer Readouts

| phase         |   layer |   auroc_mean |   auroc_std |   bal_acc_mean |
|:--------------|--------:|-------------:|------------:|---------------:|
| final_prompt  |      23 |            1 |           0 |       1        |
| final_prompt  |      22 |            1 |           0 |       0.995833 |
| final_prompt  |      21 |            1 |           0 |       1        |
| final_prompt  |      20 |            1 |           0 |       0.991667 |
| post_selector |      32 |            1 |           0 |       0.995833 |
| post_selector |      33 |            1 |           0 |       0.991667 |
| post_selector |      26 |            1 |           0 |       1        |
| post_selector |      27 |            1 |           0 |       0.995833 |

## Lexical Controls

| model_kind           |   auroc_mean |   auroc_std |   bal_acc_mean |
|:---------------------|-------------:|------------:|---------------:|
| objective_pair_tfidf |          0.5 |           0 |            0.5 |
| output_tfidf         |          0.5 |           0 |            0.5 |
| prompt_tfidf         |          0.5 |           0 |            0.5 |
| selector_tfidf       |          0.5 |           0 |            0.5 |

## Shared Counterfactual Directions

| phase          |   layer |   n_pairs |   delta_norm_mean |   direction_coherence |   heldout_cosine_mean |   heldout_positive_rate |   permutation_p |
|:---------------|--------:|----------:|------------------:|----------------------:|----------------------:|------------------------:|----------------:|
| response_mean  |      35 |       128 |           10.2274 |              0.764873 |              0.755595 |                0.984375 |      0.00497512 |
| final_prompt   |      28 |       128 |           30.0485 |              0.750712 |              0.743257 |                1        |      0.00497512 |
| response_first |      35 |       128 |           13.3105 |              0.749423 |              0.743112 |                0.96875  |      0.00497512 |
| final_prompt   |      27 |       128 |           26.1009 |              0.7502   |              0.743015 |                1        |      0.00497512 |
| final_prompt   |      26 |       128 |           23.0811 |              0.749346 |              0.742284 |                1        |      0.00497512 |
| final_prompt   |      25 |       128 |           19.7522 |              0.748238 |              0.741581 |                1        |      0.00497512 |
| final_prompt   |      29 |       128 |           33.3654 |              0.748628 |              0.740834 |                1        |      0.00497512 |
| final_prompt   |      30 |       128 |           37.3213 |              0.747977 |              0.740089 |                1        |      0.00497512 |

`heldout_cosine_mean` evaluates a mean harmful-minus-benign direction on objective domains excluded from that direction's construction. `permutation_p` uses random sign flips of paired deltas.

## Phase Transfer

| train_phase      | test_phase       |   auroc_mean |   auroc_std |   bal_acc_mean |
|:-----------------|:-----------------|-------------:|------------:|---------------:|
| final_prompt     | final_prompt     |     1        | 0           |       0.995139 |
| post_selector    | post_selector    |     1        | 0           |       1        |
| response_mean    | post_selector    |     1        | 0           |       0.965799 |
| response_mean    | response_mean    |     0.999884 | 0.000440466 |       0.996528 |
| response_first   | response_first   |     0.999653 | 0.00105947  |       0.984375 |
| final_prompt     | post_selector    |     0.999389 | 0.00316856  |       0.725174 |
| response_first   | response_mean    |     0.99728  | 0.00631227  |       0.802604 |
| public_task_span | public_task_span |     0.996759 | 0.00548501  |       0.957639 |
| response_mean    | response_first   |     0.995179 | 0.00667148  |       0.579688 |
| final_prompt     | response_mean    |     0.990046 | 0.0163367   |       0.63559  |
| response_mean    | final_prompt     |     0.981901 | 0.0327417   |       0.605729 |
| public_task_span | response_mean    |     0.968367 | 0.0338387   |       0.5875   |
| final_prompt     | response_first   |     0.966833 | 0.0421473   |       0.545833 |
| response_first   | post_selector    |     0.952416 | 0.0663439   |       0.552951 |
| public_task_span | final_prompt     |     0.931771 | 0.073502    |       0.539583 |
| public_task_span | post_selector    |     0.92353  | 0.0907486   |       0.81441  |
| post_selector    | response_mean    |     0.923235 | 0.0492251   |       0.560417 |
| response_first   | final_prompt     |     0.894083 | 0.17108     |       0.641319 |
| public_task_span | response_first   |     0.894072 | 0.0817726   |       0.560937 |
| response_mean    | public_task_span |     0.836932 | 0.083816    |       0.530382 |

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
