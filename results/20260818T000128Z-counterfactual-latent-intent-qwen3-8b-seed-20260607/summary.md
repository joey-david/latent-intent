# Counterfactual Latent Intent: counterfactual-latent-intent-qwen3-8b-seed-20260607

Two prompt conditions contain the same objective pair, public task, and teacher-forced response. They differ only in whether objective A or B is marked active. Which slot contains the harmful objective is randomized across pairs.

## Controls

- Model: `Qwen/Qwen3-8B`
- Dataset seed: `20260607`
- Matched counterfactual pairs: `128`
- Cross-validation group: held-out `scenario_id` objective domains
- Exact fixed-output match rate: `1.000`
- `pre_selector` is a hard negative control: pair members are byte-identical up to the A/B selector.

## Held-out-domain decoding by phase

The all-layer mean is reported for context, but the main quantity is the best phase-layer readout and how broadly high decodability extends across layers.

| phase            |   auroc_all_layers_mean |   best_layer |   best_layer_auroc_mean |   best_layer_auroc_std |   best_layer_bal_acc_mean |   layers_ge_0_90 |   layers_ge_0_99 |
|:-----------------|------------------------:|-------------:|------------------------:|-----------------------:|--------------------------:|-----------------:|-----------------:|
| pre_selector     |                0.5      |            0 |                0.5      |            0           |                  0.5      |                0 |                0 |
| post_selector    |                0.890026 |           16 |                1        |            0           |                  1        |               25 |               23 |
| public_task_span |                0.789648 |           22 |                0.999653 |            0.000776412 |                  0.983333 |               22 |                7 |
| final_prompt     |                0.781181 |           21 |                1        |            0           |                  1        |               19 |               18 |
| response_first   |                0.806588 |           30 |                1        |            0           |                  1        |               19 |               17 |
| response_mean    |                0.80087  |           21 |                1        |            0           |                  1        |               18 |               18 |

## Best Phase-Layer Readouts

| phase         |   layer |   auroc_mean |   auroc_std |   bal_acc_mean |
|:--------------|--------:|-------------:|------------:|---------------:|
| final_prompt  |      23 |            1 |           0 |       0.991667 |
| final_prompt  |      22 |            1 |           0 |       0.995833 |
| final_prompt  |      21 |            1 |           0 |       1        |
| final_prompt  |      20 |            1 |           0 |       0.995833 |
| post_selector |      32 |            1 |           0 |       0.991667 |
| post_selector |      33 |            1 |           0 |       0.991667 |
| post_selector |      34 |            1 |           0 |       0.991667 |
| post_selector |      27 |            1 |           0 |       1        |

## Lexical Controls

| model_kind               |   auroc_mean |   auroc_std |   bal_acc_mean |
|:-------------------------|-------------:|------------:|---------------:|
| objective_pair_tfidf     |     0.5      |    0        |       0.5      |
| output_tfidf             |     0.5      |    0        |       0.5      |
| prompt_tfidf             |     0.5      |    0        |       0.5      |
| selected_objective_tfidf |     0.672222 |    0.139443 |       0.641667 |
| selector_tfidf           |     0.5      |    0        |       0.5      |

`selected_objective_tfidf` is the strongest lexical control: it uses the selector to parse the active objective before classification, so it can express the selector/objective interaction that a bag-of-words model over the whole prompt cannot.

## Shared Counterfactual Directions

| phase          |   layer |   n_pairs |   n_scenarios |   delta_norm_mean |   direction_coherence |   heldout_cosine_mean |   heldout_positive_rate |   permutation_p | null_unit   |   null_draws |
|:---------------|--------:|----------:|--------------:|------------------:|----------------------:|----------------------:|------------------------:|----------------:|:------------|-------------:|
| response_mean  |      35 |       128 |            16 |           10.343  |              0.76804  |              0.758823 |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      27 |       128 |            16 |           26.4942 |              0.753114 |              0.745996 |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      28 |       128 |            16 |           30.5205 |              0.75325  |              0.745844 |                1        |     1.52588e-05 | scenario_id |        65536 |
| response_first |      35 |       128 |            16 |           13.2997 |              0.751062 |              0.744791 |                0.976562 |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      26 |       128 |            16 |           23.4178 |              0.751318 |              0.744304 |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      25 |       128 |            16 |           20.0389 |              0.75051  |              0.743829 |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      29 |       128 |            16 |           33.9083 |              0.751256 |              0.74361  |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      30 |       128 |            16 |           37.9476 |              0.751152 |              0.743485 |                1        |     1.52588e-05 | scenario_id |        65536 |

`heldout_cosine_mean` evaluates a mean harmful-minus-benign direction on objective domains excluded from that direction's construction. `permutation_p` uses grouped sign flips at the held-out `scenario_id` level, preserving the dependence among pairs evaluated with the same leave-one-domain-out direction. For 16 domains the 2^16 grouped sign assignments are enumerated exactly.

## Phase Transfer

| train_phase      | test_phase       |   auroc_mean |   auroc_std |   bal_acc_mean |
|:-----------------|:-----------------|-------------:|------------:|---------------:|
| final_prompt     | final_prompt     |     1        |  0          |       0.99375  |
| post_selector    | post_selector    |     1        |  0          |       1        |
| response_mean    | post_selector    |     1        |  0          |       0.963021 |
| response_mean    | response_mean    |     0.999826 |  0.00069892 |       0.996528 |
| final_prompt     | post_selector    |     0.998821 |  0.00335891 |       0.693403 |
| response_first   | response_first   |     0.998611 |  0.00342399 |       0.981424 |
| response_first   | response_mean    |     0.998322 |  0.00416575 |       0.910764 |
| public_task_span | public_task_span |     0.996933 |  0.00794518 |       0.951215 |
| response_mean    | final_prompt     |     0.995927 |  0.00600804 |       0.59184  |
| response_mean    | response_first   |     0.995265 |  0.00646225 |       0.680035 |
| final_prompt     | response_mean    |     0.993056 |  0.0100614  |       0.637153 |
| final_prompt     | response_first   |     0.975224 |  0.0365637  |       0.519792 |
| public_task_span | response_mean    |     0.967683 |  0.0300102  |       0.577778 |
| public_task_span | final_prompt     |     0.930689 |  0.0560785  |       0.550694 |
| public_task_span | post_selector    |     0.928559 |  0.100451   |       0.813194 |
| post_selector    | response_mean    |     0.923018 |  0.0498075  |       0.565104 |
| public_task_span | response_first   |     0.911303 |  0.0552744  |       0.548264 |
| response_first   | post_selector    |     0.901165 |  0.128332   |       0.621354 |
| response_mean    | public_task_span |     0.830382 |  0.0910356  |       0.527604 |
| response_first   | final_prompt     |     0.798785 |  0.231548   |       0.577083 |

## Active-vs-Inactive Objective Attention

| query_phase   |   layer |   head |   mean_active_minus_inactive |   cohens_d_vs_zero |   abs_cohens_d |   active_preference_rate |   harmful_mean |   benign_mean |   n |
|:--------------|--------:|-------:|-----------------------------:|-------------------:|---------------:|-------------------------:|---------------:|--------------:|----:|
| post_selector |       9 |     11 |                   0.0076595  |            2.42065 |        2.42065 |                 1        |     0.00796947 |    0.00734954 |  64 |
| post_selector |      14 |     11 |                   0.0107834  |            1.93857 |        1.93857 |                 0.984375 |     0.00967628 |    0.0118905  |  64 |
| post_selector |      18 |     13 |                  -0.0106195  |           -1.92734 |        1.92734 |                 0        |    -0.00944439 |   -0.0117946  |  64 |
| post_selector |      18 |     12 |                  -0.0110088  |           -1.89379 |        1.89379 |                 0        |    -0.012034   |   -0.00998366 |  64 |
| post_selector |      19 |     30 |                  -0.0270359  |           -1.88589 |        1.88589 |                 0        |    -0.0321833  |   -0.0218884  |  64 |
| post_selector |      13 |     14 |                   0.0192039  |            1.77191 |        1.77191 |                 1        |     0.0192352  |    0.0191727  |  64 |
| post_selector |      24 |     21 |                   0.00308776 |            1.68377 |        1.68377 |                 1        |     0.00268083 |    0.0034947  |  64 |
| post_selector |      21 |     31 |                   0.0105992  |            1.66158 |        1.66158 |                 0.984375 |     0.00967971 |    0.0115188  |  64 |
| post_selector |      16 |     23 |                   0.0465764  |            1.57648 |        1.57648 |                 0.921875 |     0.0381337  |    0.0550191  |  64 |
| post_selector |      20 |      9 |                   0.00205855 |            1.56959 |        1.56959 |                 0.921875 |     0.00184358 |    0.00227351 |  64 |
| post_selector |      15 |      5 |                   0.00381766 |            1.48856 |        1.48856 |                 0.96875  |     0.00353753 |    0.00409779 |  64 |
| post_selector |      20 |     15 |                   0.0339376  |            1.48145 |        1.48145 |                 0.90625  |     0.0285676  |    0.0393076  |  64 |
| post_selector |      18 |     15 |                   0.0309179  |            1.47705 |        1.47705 |                 1        |     0.0233054  |    0.0385303  |  64 |
| post_selector |      17 |     15 |                   0.0429358  |            1.44758 |        1.44758 |                 1        |     0.0344858  |    0.0513858  |  64 |
| post_selector |      19 |      0 |                  -0.00980077 |           -1.40304 |        1.40304 |                 0.03125  |    -0.010159   |   -0.00944253 |  64 |
| post_selector |      21 |     18 |                   0.0112886  |            1.38078 |        1.38078 |                 0.984375 |     0.0134636  |    0.00911371 |  64 |
| post_selector |      16 |     17 |                  -0.0218575  |           -1.37206 |        1.37206 |                 0        |    -0.0239216  |   -0.0197935  |  64 |
| post_selector |      17 |      4 |                   0.0226051  |            1.36248 |        1.36248 |                 1        |     0.014942   |    0.0302682  |  64 |
| post_selector |       9 |      8 |                   0.00204767 |            1.35332 |        1.35332 |                 0.9375   |     0.00214076 |    0.00195457 |  64 |
| post_selector |       9 |     10 |                   0.0100144  |            1.34984 |        1.34984 |                 1        |     0.0115301  |    0.00849875 |  64 |

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
