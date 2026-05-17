from pathlib import Path

import numpy as np

from latent_intent_probe.config import ProbeConfig
from latent_intent_probe.probes import summarize_paired_directions


def test_paired_direction_is_zero_before_selector_and_shared_after(tmp_path: Path) -> None:
    phases = np.array(["pre_selector", "post_selector"])
    records = []
    activations = []

    for pair_idx in range(8):
        scenario = f"scenario-{pair_idx % 4}"
        pair_id = f"pair-{pair_idx}"
        benign = np.zeros((2, 1, 3), dtype=np.float32)
        harmful = np.zeros((2, 1, 3), dtype=np.float32)
        harmful[1, 0] = np.array([1.0, 0.2, 0.0], dtype=np.float32)

        records.append({"pair_id": pair_id, "label": 0, "scenario_id": scenario})
        activations.append(benign)
        records.append({"pair_id": pair_id, "label": 1, "scenario_id": scenario})
        activations.append(harmful)

    path = tmp_path / "activations.npz"
    np.savez_compressed(path, activations=np.stack(activations), phase_names=phases)

    summary = summarize_paired_directions(
        path,
        records,
        ProbeConfig(permutation_samples=50),
        seed=1,
    )

    pre = summary[(summary["phase"] == "pre_selector") & (summary["layer"] == 0)].iloc[0]
    post = summary[(summary["phase"] == "post_selector") & (summary["layer"] == 0)].iloc[0]

    assert pre["direction_coherence"] == 0.0
    assert pre["heldout_cosine_mean"] == 0.0
    assert post["direction_coherence"] > 0.99
    assert post["heldout_cosine_mean"] > 0.99
    assert post["heldout_positive_rate"] == 1.0

    # Four held-out domains -> exact grouped null has 2^4 assignments. Because
    # every domain points in the same direction, only the all-positive assignment
    # reaches the observed statistic.
    assert post["null_unit"] == "scenario_id"
    assert post["n_scenarios"] == 4
    assert post["null_draws"] == 16
    assert np.isclose(post["permutation_p"], 1 / 16)
