# Counterfactual Latent Intent: counterfactual-latent-intent-qwen3-8b-seed-20260818

Two prompt conditions contain the same objective pair, public task, and teacher-forced response. They differ only in whether objective A or B is marked active. Which slot contains the harmful objective is randomized across pairs.

## Controls

- Model: `Qwen/Qwen3-8B`
- Dataset seed: `20260818`
- Matched counterfactual pairs: `128`
- Cross-validation group: held-out `scenario_id` objective domains
- Exact fixed-output match rate: `1.000`
- `pre_selector` is a hard negative control: pair members are byte-identical up to the A/B selector.

## Held-out-domain decoding by phase

The all-layer mean is reported for context, but the main quantity is the best phase-layer readout and how broadly high decodability extends across layers.

| phase            |   auroc_all_layers_mean |   best_layer |   best_layer_auroc_mean |   best_layer_auroc_std |   best_layer_bal_acc_mean |   layers_ge_0_90 |   layers_ge_0_99 |
|:-----------------|------------------------:|-------------:|------------------------:|-----------------------:|--------------------------:|-----------------:|-----------------:|
| pre_selector     |                0.5      |            0 |                     0.5 |                      0 |                  0.5      |                0 |                0 |
| post_selector    |                0.869041 |           16 |                     1   |                      0 |                  1        |               25 |               23 |
| public_task_span |                0.797078 |           21 |                     1   |                      0 |                  0.980208 |               21 |               13 |
| final_prompt     |                0.773349 |           19 |                     1   |                      0 |                  0.995833 |               19 |               18 |
| response_first   |                0.791382 |           31 |                     1   |                      0 |                  0.995833 |               19 |               17 |
| response_mean    |                0.795555 |           21 |                     1   |                      0 |                  0.992708 |               19 |               18 |

## Best Phase-Layer Readouts

| phase         |   layer |   auroc_mean |   auroc_std |   bal_acc_mean |
|:--------------|--------:|-------------:|------------:|---------------:|
| final_prompt  |      23 |            1 |           0 |       0.986458 |
| final_prompt  |      22 |            1 |           0 |       0.989583 |
| final_prompt  |      21 |            1 |           0 |       0.991667 |
| final_prompt  |      19 |            1 |           0 |       0.995833 |
| final_prompt  |      20 |            1 |           0 |       0.991667 |
| post_selector |      33 |            1 |           0 |       1        |
| post_selector |      34 |            1 |           0 |       1        |
| post_selector |      35 |            1 |           0 |       1        |

## Lexical Controls

| model_kind               |   auroc_mean |   auroc_std |   bal_acc_mean |
|:-------------------------|-------------:|------------:|---------------:|
| objective_pair_tfidf     |     0.5      |    0        |       0.5      |
| output_tfidf             |     0.5      |    0        |       0.5      |
| prompt_tfidf             |     0.5      |    0        |       0.5      |
| selected_objective_tfidf |     0.652083 |    0.122279 |       0.658333 |
| selector_tfidf           |     0.5      |    0        |       0.5      |

`selected_objective_tfidf` is the strongest lexical control: it uses the selector to parse the active objective before classification, so it can express the selector/objective interaction that a bag-of-words model over the whole prompt cannot.

## Shared Counterfactual Directions

| phase          |   layer |   n_pairs |   n_scenarios |   delta_norm_mean |   direction_coherence |   heldout_cosine_mean |   heldout_positive_rate |   permutation_p | null_unit   |   null_draws |
|:---------------|--------:|----------:|--------------:|------------------:|----------------------:|----------------------:|------------------------:|----------------:|:------------|-------------:|
| response_first |      35 |       128 |            16 |           13.8078 |              0.794242 |              0.78894  |                0.976562 |     1.52588e-05 | scenario_id |        65536 |
| response_mean  |      35 |       128 |            16 |           10.3619 |              0.784447 |              0.77601  |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      25 |       128 |            16 |           19.7504 |              0.745961 |              0.738933 |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      28 |       128 |            16 |           30.0877 |              0.746692 |              0.73883  |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      27 |       128 |            16 |           26.1305 |              0.746308 |              0.738758 |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      26 |       128 |            16 |           23.1062 |              0.745863 |              0.738486 |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      29 |       128 |            16 |           33.4433 |              0.74481  |              0.736631 |                1        |     1.52588e-05 | scenario_id |        65536 |
| final_prompt   |      30 |       128 |            16 |           37.4407 |              0.744137 |              0.735861 |                1        |     1.52588e-05 | scenario_id |        65536 |

`heldout_cosine_mean` evaluates a mean harmful-minus-benign direction on objective domains excluded from that direction's construction. `permutation_p` uses grouped sign flips at the held-out `scenario_id` level, preserving the dependence among pairs evaluated with the same leave-one-domain-out direction. For 16 domains the 2^16 grouped sign assignments are enumerated exactly.

## Phase Transfer

| train_phase      | test_phase       |   auroc_mean |   auroc_std |   bal_acc_mean |
|:-----------------|:-----------------|-------------:|------------:|---------------:|
| post_selector    | post_selector    |     1        | 0           |       1        |
| final_prompt     | final_prompt     |     0.999837 | 0.000632538 |       0.987674 |
| response_first   | response_first   |     0.999819 | 0.00071655  |       0.982292 |
| response_mean    | response_mean    |     0.999805 | 0.00106977  |       0.987674 |
| response_first   | response_mean    |     0.999291 | 0.00286356  |       0.848264 |
| public_task_span | public_task_span |     0.999266 | 0.00200253  |       0.962326 |
| response_mean    | post_selector    |     0.999154 | 0.00249336  |       0.775174 |
| final_prompt     | post_selector    |     0.995804 | 0.00969475  |       0.635764 |
| final_prompt     | response_mean    |     0.994564 | 0.00898748  |       0.507465 |
| response_mean    | response_first   |     0.993783 | 0.0055664   |       0.641493 |
| final_prompt     | response_first   |     0.987015 | 0.0176119   |       0.503993 |
| response_mean    | final_prompt     |     0.983131 | 0.0490643   |       0.58941  |
| public_task_span | post_selector    |     0.982292 | 0.039683    |       0.709549 |
| public_task_span | response_mean    |     0.97989  | 0.0219447   |       0.5      |
| public_task_span | final_prompt     |     0.966189 | 0.0246143   |       0.5      |
| post_selector    | response_mean    |     0.941095 | 0.0478527   |       0.584028 |
| response_first   | final_prompt     |     0.940271 | 0.131674    |       0.666146 |
| response_first   | post_selector    |     0.937254 | 0.0631277   |       0.607118 |
| public_task_span | response_first   |     0.932288 | 0.0501655   |       0.5      |
| post_selector    | final_prompt     |     0.813715 | 0.144296    |       0.506944 |

## Active-vs-Inactive Objective Attention

| query_phase   |   layer |   head |   mean_active_minus_inactive |   cohens_d_vs_zero |   abs_cohens_d |   active_preference_rate |   harmful_mean |   benign_mean |   n |
|:--------------|--------:|-------:|-----------------------------:|-------------------:|---------------:|-------------------------:|---------------:|--------------:|----:|
| post_selector |       9 |     11 |                   0.00784826 |            2.38168 |        2.38168 |                 1        |     0.00784835 |    0.00784818 |  64 |
| post_selector |      19 |     30 |                  -0.0306905  |           -2.15909 |        2.15909 |                 0        |    -0.0383677  |   -0.0230134  |  64 |
| post_selector |      14 |     11 |                   0.0118946  |            2.11617 |        2.11617 |                 1        |     0.0112857  |    0.0125035  |  64 |
| post_selector |      18 |     12 |                  -0.0118715  |           -2.04016 |        2.04016 |                 0        |    -0.0136503  |   -0.0100928  |  64 |
| post_selector |      16 |     17 |                  -0.0265296  |           -1.97398 |        1.97398 |                 0        |    -0.0312452  |   -0.0218141  |  64 |
| post_selector |      16 |     23 |                   0.0533538  |            1.94408 |        1.94408 |                 1        |     0.0468858  |    0.0598219  |  64 |
| post_selector |      17 |     15 |                   0.0497985  |            1.77706 |        1.77706 |                 1        |     0.0423635  |    0.0572334  |  64 |
| post_selector |      18 |     15 |                   0.0331906  |            1.77175 |        1.77175 |                 1        |     0.0242367  |    0.0421445  |  64 |
| post_selector |      18 |     13 |                  -0.0107801  |           -1.75031 |        1.75031 |                 0        |    -0.0109955  |   -0.0105647  |  64 |
| post_selector |      13 |     14 |                   0.0214248  |            1.57498 |        1.57498 |                 1        |     0.0212852  |    0.0215643  |  64 |
| post_selector |      17 |      4 |                   0.0248382  |            1.56558 |        1.56558 |                 1        |     0.0175592  |    0.0321172  |  64 |
| post_selector |      18 |     11 |                  -0.0298193  |           -1.55324 |        1.55324 |                 0.015625 |    -0.0392229  |   -0.0204157  |  64 |
| post_selector |      21 |     31 |                   0.0111368  |            1.55165 |        1.55165 |                 0.96875  |     0.00981348 |    0.0124601  |  64 |
| post_selector |      24 |     21 |                   0.00345957 |            1.54178 |        1.54178 |                 0.96875  |     0.00442269 |    0.00249645 |  64 |
| post_selector |      20 |     15 |                   0.0374139  |            1.52596 |        1.52596 |                 0.96875  |     0.0270267  |    0.0478012  |  64 |
| post_selector |      15 |      5 |                   0.00397964 |            1.49191 |        1.49191 |                 0.953125 |     0.00289146 |    0.00506783 |  64 |
| post_selector |      21 |     18 |                   0.0115459  |            1.4076  |        1.4076  |                 0.953125 |     0.0124298  |    0.0106619  |  64 |
| post_selector |      26 |     24 |                   0.00288233 |            1.3753  |        1.3753  |                 0.921875 |     0.00244403 |    0.00332063 |  64 |
| post_selector |      19 |      0 |                  -0.0107186  |           -1.34742 |        1.34742 |                 0.078125 |    -0.0126309  |   -0.00880629 |  64 |
| post_selector |      15 |      7 |                   0.0459114  |            1.34323 |        1.34323 |                 0.921875 |     0.0270631  |    0.0647598  |  64 |

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
