# Counterfactual Latent Intent: counterfactual-latent-intent-qwen3-8b-seed-20260819

Two prompt conditions contain the same objective pair, public task, and teacher-forced response. They differ only in whether objective A or B is marked active. Which slot contains the harmful objective is randomized across pairs.

## Controls

- Model: `Qwen/Qwen3-8B`
- Dataset seed: `20260819`
- Matched counterfactual pairs: `128`
- Cross-validation group: held-out `scenario_id` objective domains
- Exact fixed-output match rate: `1.000`
- `pre_selector` is a hard negative control: pair members are byte-identical up to the A/B selector.

## Held-out-domain decoding by phase

The all-layer mean is reported for context, but the main quantity is the best phase-layer readout and how broadly high decodability extends across layers.

| phase            |   auroc_all_layers_mean |   best_layer |   best_layer_auroc_mean |   best_layer_auroc_std |   best_layer_bal_acc_mean |   layers_ge_0_90 |   layers_ge_0_99 |
|:-----------------|------------------------:|-------------:|------------------------:|-----------------------:|--------------------------:|-----------------:|-----------------:|
| pre_selector     |                0.5      |            0 |                0.5      |             0          |                  0.5      |                0 |                0 |
| post_selector    |                0.881847 |           16 |                1        |             0          |                  1        |               25 |               23 |
| public_task_span |                0.790334 |           22 |                0.996181 |             0.00854054 |                  0.9625   |               21 |                6 |
| final_prompt     |                0.777764 |           21 |                1        |             0          |                  0.991667 |               19 |               18 |
| response_first   |                0.7898   |           29 |                1        |             0          |                  0.995833 |               19 |               17 |
| response_mean    |                0.795917 |           21 |                1        |             0          |                  0.995833 |               20 |               18 |

## Best Phase-Layer Readouts

| phase         |   layer |   auroc_mean |   auroc_std |   bal_acc_mean |
|:--------------|--------:|-------------:|------------:|---------------:|
| final_prompt  |      21 |            1 |           0 |       0.991667 |
| final_prompt  |      22 |            1 |           0 |       0.979167 |
| final_prompt  |      23 |            1 |           0 |       0.983333 |
| post_selector |      31 |            1 |           0 |       0.996875 |
| post_selector |      32 |            1 |           0 |       0.996875 |
| post_selector |      33 |            1 |           0 |       0.996875 |
| post_selector |      34 |            1 |           0 |       0.996875 |
| post_selector |      35 |            1 |           0 |       0.996875 |

## Lexical Controls

| model_kind               |   auroc_mean |   auroc_std |   bal_acc_mean |
|:-------------------------|-------------:|------------:|---------------:|
| objective_pair_tfidf     |     0.5      |    0        |       0.5      |
| output_tfidf             |     0.5      |    0        |       0.5      |
| prompt_tfidf             |     0.5      |    0        |       0.5      |
| selected_objective_tfidf |     0.655556 |    0.132637 |       0.633333 |
| selector_tfidf           |     0.5      |    0        |       0.5      |

`selected_objective_tfidf` is the strongest lexical control: it uses the selector to parse the active objective before classification, so it can express the selector/objective interaction that a bag-of-words model over the whole prompt cannot.

## Shared Counterfactual Directions

| phase          |   layer |   n_pairs |   n_scenarios |   delta_norm_mean |   direction_coherence |   heldout_cosine_mean |   heldout_positive_rate |   permutation_p | null_unit   |   null_draws |
|:---------------|--------:|----------:|--------------:|------------------:|----------------------:|----------------------:|------------------------:|----------------:|:------------|-------------:|
| response_first |      35 |       128 |            16 |           13.4117 |              0.782096 |              0.775879 |                0.984375 |     1.52588e-05 | scenario_id |        65536 |
| response_mean  |      35 |       128 |            16 |           10.0559 |              0.772071 |              0.762931 |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      27 |       128 |            16 |           25.751  |              0.748818 |              0.74146  |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      28 |       128 |            16 |           29.6462 |              0.749004 |              0.74136  |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      26 |       128 |            16 |           22.7925 |              0.747926 |              0.74071  |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      25 |       128 |            16 |           19.4802 |              0.747418 |              0.740495 |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      30 |       128 |            16 |           36.8986 |              0.747311 |              0.739342 |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      29 |       128 |            16 |           32.9367 |              0.747206 |              0.7393   |                1        |     1.52588e-05 | scenario_id |        65536 |

`heldout_cosine_mean` evaluates a mean harmful-minus-benign direction on objective domains excluded from that direction's construction. `permutation_p` uses grouped sign flips at the held-out `scenario_id` level, preserving the dependence among pairs evaluated with the same leave-one-domain-out direction. For 16 domains the 2^16 grouped sign assignments are enumerated exactly.

## Phase Transfer

| train_phase      | test_phase       |   auroc_mean |   auroc_std |   bal_acc_mean |
|:-----------------|:-----------------|-------------:|------------:|---------------:|
| final_prompt     | final_prompt     |     1        | 0           |       0.980556 |
| post_selector    | post_selector    |     1        | 0           |       1        |
| response_mean    | response_mean    |     0.999884 | 0.000633938 |       0.990972 |
| response_mean    | post_selector    |     0.999479 | 0.00285272  |       0.801736 |
| response_first   | response_first   |     0.9989   | 0.0031417   |       0.983681 |
| final_prompt     | post_selector    |     0.996376 | 0.00931311  |       0.761806 |
| public_task_span | public_task_span |     0.993287 | 0.0140875   |       0.953819 |
| response_first   | response_mean    |     0.993197 | 0.0220373   |       0.711806 |
| final_prompt     | response_mean    |     0.99256  | 0.0130133   |       0.527083 |
| public_task_span | post_selector    |     0.992336 | 0.0112761   |       0.693576 |
| response_mean    | response_first   |     0.989276 | 0.016044    |       0.648958 |
| public_task_span | response_mean    |     0.976313 | 0.0215942   |       0.5      |
| final_prompt     | response_first   |     0.974736 | 0.0347636   |       0.500694 |
| public_task_span | final_prompt     |     0.949468 | 0.0562375   |       0.5      |
| post_selector    | response_mean    |     0.918334 | 0.0490663   |       0.698958 |
| public_task_span | response_first   |     0.908019 | 0.0702142   |       0.5      |
| response_mean    | final_prompt     |     0.900514 | 0.154066    |       0.516319 |
| response_first   | final_prompt     |     0.841905 | 0.204715    |       0.592708 |
| response_first   | post_selector    |     0.831049 | 0.191086    |       0.552257 |
| post_selector    | final_prompt     |     0.787001 | 0.133478    |       0.545833 |

## Active-vs-Inactive Objective Attention

| query_phase   |   layer |   head |   mean_active_minus_inactive |   cohens_d_vs_zero |   abs_cohens_d |   active_preference_rate |   harmful_mean |   benign_mean |   n |
|:--------------|--------:|-------:|-----------------------------:|-------------------:|---------------:|-------------------------:|---------------:|--------------:|----:|
| post_selector |       9 |     11 |                   0.00764615 |            2.82724 |        2.82724 |                 1        |     0.00739541 |    0.00789689 |  64 |
| post_selector |      14 |     11 |                   0.0113878  |            2.05219 |        2.05219 |                 0.984375 |     0.0102341  |    0.0125414  |  64 |
| post_selector |      18 |     13 |                  -0.0107903  |           -1.94451 |        1.94451 |                 0        |    -0.0101807  |   -0.0113999  |  64 |
| post_selector |      21 |     31 |                   0.0112823  |            1.89314 |        1.89314 |                 0.984375 |     0.011773   |    0.0107917  |  64 |
| post_selector |      18 |     12 |                  -0.0105052  |           -1.89124 |        1.89124 |                 0.015625 |    -0.0119411  |   -0.00906929 |  64 |
| post_selector |       9 |     10 |                   0.00962206 |            1.87469 |        1.87469 |                 1        |     0.0101474  |    0.00909674 |  64 |
| post_selector |      19 |     30 |                  -0.027417   |           -1.84928 |        1.84928 |                 0        |    -0.0361455  |   -0.0186886  |  64 |
| post_selector |      15 |      5 |                   0.00405219 |            1.69562 |        1.69562 |                 1        |     0.003405   |    0.00469937 |  64 |
| post_selector |      24 |     21 |                   0.00340147 |            1.67411 |        1.67411 |                 1        |     0.00343918 |    0.00336376 |  64 |
| post_selector |      19 |      0 |                  -0.0103095  |           -1.67178 |        1.67178 |                 0.03125  |    -0.0117529  |   -0.00886615 |  64 |
| post_selector |      21 |     18 |                   0.0115973  |            1.66165 |        1.66165 |                 1        |     0.014618   |    0.00857667 |  64 |
| post_selector |      13 |     14 |                   0.0220382  |            1.65855 |        1.65855 |                 1        |     0.0238409  |    0.0202355  |  64 |
| post_selector |      16 |     23 |                   0.0470283  |            1.46144 |        1.46144 |                 0.984375 |     0.0394914  |    0.0545653  |  64 |
| post_selector |      18 |     15 |                   0.0315451  |            1.4439  |        1.4439  |                 1        |     0.0212881  |    0.0418021  |  64 |
| post_selector |      16 |     17 |                  -0.0235947  |           -1.44359 |        1.44359 |                 0        |    -0.0281883  |   -0.019001   |  64 |
| post_selector |      18 |     11 |                  -0.0262136  |           -1.42898 |        1.42898 |                 0.046875 |    -0.0328729  |   -0.0195544  |  64 |
| post_selector |      17 |     15 |                   0.044154   |            1.42729 |        1.42729 |                 1        |     0.0347745  |    0.0535336  |  64 |
| post_selector |      20 |      9 |                   0.00205313 |            1.42668 |        1.42668 |                 0.921875 |     0.0018577  |    0.00224855 |  64 |
| post_selector |      20 |     15 |                   0.0379608  |            1.40137 |        1.40137 |                 0.984375 |     0.029629   |    0.0462927  |  64 |
| post_selector |      19 |     25 |                   0.00607661 |            1.38508 |        1.38508 |                 0.921875 |     0.0040562  |    0.00809702 |  64 |

## Main Artifacts

- `counterfactual_result.png`: held-out decoding and paired-direction heatmaps
- `phase_probe_summary.csv`: best-layer and across-layer phase summaries
- `paired_direction_summary.csv`: counterfactual direction statistics
- `phase_transfer_summary.csv`: train-phase to test-phase generalization
- `attention_asymmetry_summary.csv`: active-vs-inactive objective attention heads
- `text_baseline_summary.csv`: lexical controls, including selected-objective TF-IDF
- `output_control.csv`: verifies the public response is identical

## Interpretation

This experiment studies a synthetic, prompt-injected selected objective, not autonomous malicious intent. The interesting evidence is whether a label that is unreadable before the selector becomes readable afterward, generalizes to held-out objective domains, persists into an identical response, and yields a shared paired activation direction or active-objective attention preference.
