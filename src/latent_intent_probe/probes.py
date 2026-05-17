from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from latent_intent_probe.config import ProbeConfig


def fit_activation_probes(
    activations_path: str | Path,
    records: list[dict],
    config: ProbeConfig,
    seed: int,
) -> pd.DataFrame:
    activations, phase_names = _load_activations(activations_path)
    labels = np.array([record["label"] for record in records])
    groups = np.array([record["scenario_id"] for record in records])

    rows = []
    total = len(phase_names) * activations.shape[2]
    with tqdm(total=total, desc="activation probes", unit="phase-layer") as progress:
        for phase_idx, phase in enumerate(phase_names):
            for layer in range(activations.shape[2]):
                rows.extend(
                    _cross_validated_scores(
                        activations[:, phase_idx, layer, :],
                        labels,
                        groups,
                        config,
                        seed,
                        model_kind="phase_activation",
                        layer=layer,
                        phase=phase,
                    )
                )
                progress.update(1)
    return pd.DataFrame(rows)


def fit_phase_transfer_probes(
    activations_path: str | Path,
    records: list[dict],
    activation_metrics: pd.DataFrame,
    config: ProbeConfig,
    seed: int,
) -> pd.DataFrame:
    activations, phase_names = _load_activations(activations_path)
    labels = np.array([record["label"] for record in records])
    groups = np.array([record["scenario_id"] for record in records])
    top_layers = _top_layers(activation_metrics, config.transfer_top_k_layers)

    rows = []
    folds = list(_splitter(labels, groups, config, seed).split(activations, labels, groups))
    total = len(top_layers) * len(phase_names) * len(phase_names)
    with tqdm(total=total, desc="phase transfer", unit="phase-pair") as progress:
        for layer in top_layers:
            for train_phase_idx, train_phase in enumerate(phase_names):
                for test_phase_idx, test_phase in enumerate(phase_names):
                    for fold, (train_idx, test_idx) in enumerate(folds):
                        clf = _activation_clf(config, seed)
                        clf.fit(activations[train_idx, train_phase_idx, layer, :], labels[train_idx])
                        probabilities = clf.predict_proba(
                            activations[test_idx, test_phase_idx, layer, :]
                        )[:, 1]
                        predictions = (probabilities >= 0.5).astype(int)
                        row = _score_row(
                            labels[test_idx],
                            predictions,
                            probabilities,
                            model_kind="phase_transfer",
                            fold=fold,
                            layer=layer,
                            phase=test_phase,
                        )
                        row["train_phase"] = train_phase
                        row["test_phase"] = test_phase
                        rows.append(row)
                    progress.update(1)
    return pd.DataFrame(rows)


def fit_text_baselines(records: list[dict], config: ProbeConfig, seed: int) -> pd.DataFrame:
    labels = np.array([record["label"] for record in records])
    groups = np.array([record["scenario_id"] for record in records])
    baselines = {
        "output_tfidf": [record.get("response", "") for record in records],
        "prompt_tfidf": [record.get("rendered_prompt", "") for record in records],
        "selector_tfidf": [f"active_slot_{record['active_slot']}" for record in records],
        "objective_pair_tfidf": [
            f"{record['objective_a']} || {record['objective_b']}" for record in records
        ],
        # Stronger lexical control: use the selector to parse out the selected
        # objective before fitting TF-IDF. Unlike prompt_tfidf, this baseline can
        # express the selector/objective interaction directly.
        "selected_objective_tfidf": [record["private_objective"] for record in records],
    }

    rows = []
    for name, texts in tqdm(
        baselines.items(),
        total=len(baselines),
        desc="text baselines",
        unit="baseline",
    ):
        rows.extend(_cross_validated_text_scores(texts, labels, groups, config, seed, name))
    return pd.DataFrame(rows)


def summarize_paired_directions(
    activations_path: str | Path,
    records: list[dict],
    config: ProbeConfig,
    seed: int,
) -> pd.DataFrame:
    """Measure whether harmful-minus-benign pair deltas share a reusable direction.

    A direction is learned from all objective domains except one and evaluated on
    that held-out domain. The null sign-flips entire held-out objective domains,
    rather than individual pair scores, so pairs that share one leave-one-domain-
    out direction remain dependent under the null.
    """
    activations, phase_names = _load_activations(activations_path)
    pair_rows: dict[str, dict[int, int]] = {}
    for idx, record in enumerate(records):
        pair_rows.setdefault(record["pair_id"], {})[int(record["label"])] = idx

    deltas = []
    scenarios = []
    for label_map in pair_rows.values():
        if 0 not in label_map or 1 not in label_map:
            continue
        benign_idx = label_map[0]
        harmful_idx = label_map[1]
        deltas.append(activations[harmful_idx] - activations[benign_idx])
        scenarios.append(records[harmful_idx]["scenario_id"])

    if not deltas:
        return pd.DataFrame()

    delta_array = np.stack(deltas, axis=0)
    scenarios_array = np.array(scenarios)
    rng = np.random.default_rng(seed)
    rows = []

    total = len(phase_names) * delta_array.shape[2]
    with tqdm(total=total, desc="paired directions", unit="phase-layer") as progress:
        for phase_idx, phase in enumerate(phase_names):
            for layer in range(delta_array.shape[2]):
                x = delta_array[:, phase_idx, layer, :].astype(np.float64, copy=False)
                norms = np.linalg.norm(x, axis=1)
                valid = norms > 1e-10
                if valid.sum() < 2:
                    rows.append(
                        {
                            "phase": phase,
                            "layer": layer,
                            "n_pairs": int(valid.sum()),
                            "n_scenarios": 0,
                            "delta_norm_mean": float(norms.mean()),
                            "direction_coherence": 0.0,
                            "heldout_cosine_mean": 0.0,
                            "heldout_positive_rate": 0.0,
                            "permutation_p": 1.0,
                            "null_unit": "scenario_id",
                            "null_draws": 0,
                        }
                    )
                    progress.update(1)
                    continue

                u = x[valid] / norms[valid, None]
                valid_scenarios = scenarios_array[valid]
                direction_coherence = float(np.linalg.norm(u.mean(axis=0)))

                heldout_cosines: list[float] = []
                heldout_scenarios: list[str] = []
                for scenario in np.unique(valid_scenarios):
                    train = u[valid_scenarios != scenario]
                    test = u[valid_scenarios == scenario]
                    if len(train) == 0 or len(test) == 0:
                        continue
                    direction = train.mean(axis=0)
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm <= 1e-10:
                        continue
                    direction /= direction_norm
                    scenario_scores = test @ direction
                    heldout_cosines.extend(scenario_scores.tolist())
                    heldout_scenarios.extend([str(scenario)] * len(scenario_scores))

                scores = np.asarray(heldout_cosines, dtype=float)
                score_scenarios = np.asarray(heldout_scenarios, dtype=str)
                heldout_mean = float(scores.mean()) if len(scores) else 0.0
                heldout_positive = float((scores > 0).mean()) if len(scores) else 0.0

                if len(scores):
                    permutation_p, null_draws = _grouped_signflip_p(
                        scores,
                        score_scenarios,
                        config,
                        rng,
                    )
                    n_scenarios = len(np.unique(score_scenarios))
                else:
                    permutation_p = 1.0
                    null_draws = 0
                    n_scenarios = 0

                rows.append(
                    {
                        "phase": phase,
                        "layer": layer,
                        "n_pairs": int(valid.sum()),
                        "n_scenarios": int(n_scenarios),
                        "delta_norm_mean": float(norms[valid].mean()),
                        "direction_coherence": direction_coherence,
                        "heldout_cosine_mean": heldout_mean,
                        "heldout_positive_rate": heldout_positive,
                        "permutation_p": permutation_p,
                        "null_unit": "scenario_id",
                        "null_draws": int(null_draws),
                    }
                )
                progress.update(1)

    return pd.DataFrame(rows)


def _grouped_signflip_p(
    scores: np.ndarray,
    scenarios: np.ndarray,
    config: ProbeConfig,
    rng: np.random.Generator,
) -> tuple[float, int]:
    unique_scenarios = np.unique(scenarios)
    domain_sums = np.array(
        [scores[scenarios == scenario].sum() for scenario in unique_scenarios],
        dtype=np.float64,
    )
    observed = float(scores.mean())
    n_domains = len(unique_scenarios)

    # With 16 domains the exact null is only 2^16 = 65,536 sign assignments.
    # For larger studies, fall back to Monte Carlo grouped sign flips.
    if n_domains <= 20:
        signs = _exact_sign_matrix(n_domains)
        null_means = (signs @ domain_sums) / len(scores)
        p_value = float(np.count_nonzero(null_means >= observed - 1e-12) / len(null_means))
        return p_value, len(null_means)

    draws = max(1, int(config.permutation_samples))
    signs = rng.choice((-1.0, 1.0), size=(draws, n_domains))
    null_means = (signs @ domain_sums) / len(scores)
    p_value = float((1 + np.count_nonzero(null_means >= observed)) / (draws + 1))
    return p_value, draws


@lru_cache(maxsize=None)
def _exact_sign_matrix(n_groups: int) -> np.ndarray:
    integers = np.arange(1 << n_groups, dtype=np.uint32)[:, None]
    shifts = np.arange(n_groups, dtype=np.uint32)[None, :]
    bits = ((integers >> shifts) & 1).astype(np.float64)
    return 2.0 * bits - 1.0


def summarize_attention_asymmetry(attention_path: str | Path) -> pd.DataFrame:
    path = Path(attention_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    attention = pd.read_csv(path)
    if attention.empty or "active_objective_asymmetry" not in attention:
        return pd.DataFrame()

    rows = []
    grouped = attention.groupby(["query_phase", "layer", "head"], dropna=False)
    for (phase, layer, head), group in tqdm(
        grouped,
        total=grouped.ngroups,
        desc="attention summary",
        unit="head",
    ):
        asymmetry = group["active_objective_asymmetry"].to_numpy(dtype=float)
        if len(asymmetry) < 2:
            continue
        std = float(asymmetry.std(ddof=1))
        mean = float(asymmetry.mean())
        harmful = group[group["label"] == 1]["active_objective_asymmetry"].to_numpy(dtype=float)
        benign = group[group["label"] == 0]["active_objective_asymmetry"].to_numpy(dtype=float)
        rows.append(
            {
                "query_phase": phase,
                "layer": int(layer),
                "head": int(head),
                "mean_active_minus_inactive": mean,
                "cohens_d_vs_zero": 0.0 if std == 0 else mean / std,
                "abs_cohens_d": 0.0 if std == 0 else abs(mean / std),
                "active_preference_rate": float((asymmetry > 0).mean()),
                "harmful_mean": float(harmful.mean()) if len(harmful) else 0.0,
                "benign_mean": float(benign.mean()) if len(benign) else 0.0,
                "n": len(group),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("abs_cohens_d", ascending=False)


def summarize_output_control(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": record["id"],
                "pair_id": record["pair_id"],
                "label": record["label"],
                "response": record.get("response", ""),
                "fixed_response": record.get("fixed_response", ""),
                "matches_fixed_response": (
                    record.get("response", "") == record.get("fixed_response", "")
                ),
            }
            for record in records
        ]
    )


def _load_activations(activations_path: str | Path) -> tuple[np.ndarray, list[str]]:
    payload = np.load(activations_path)
    activations = payload["activations"]
    if activations.ndim == 3:
        return activations[:, None, :, :], ["final_prompt"]
    return activations, [str(phase) for phase in payload["phase_names"].tolist()]


def _top_layers(metrics: pd.DataFrame, count: int) -> list[int]:
    summary = (
        metrics.groupby("layer", dropna=False)
        .agg(auroc_mean=("auroc", "mean"))
        .reset_index()
        .sort_values("auroc_mean", ascending=False)
    )
    return [int(layer) for layer in summary["layer"].head(count).tolist()]


def _splitter(labels: np.ndarray, groups: np.ndarray, config: ProbeConfig, seed: int):
    n_groups = len(set(groups.tolist()))
    n_splits = min(config.cv_folds, n_groups)
    if n_splits < 2:
        raise ValueError("Need at least two scenario groups for cross-validation")
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _activation_clf(config: ProbeConfig, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.regularization_c,
            class_weight="balanced",
            max_iter=config.max_iter,
            random_state=seed,
        ),
    )


def _cross_validated_scores(
    x: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    config: ProbeConfig,
    seed: int,
    model_kind: str,
    layer: int | None,
    phase: str | None,
) -> list[dict]:
    rows = []
    splitter = _splitter(labels, groups, config, seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, labels, groups)):
        clf = _activation_clf(config, seed)
        clf.fit(x[train_idx], labels[train_idx])
        probabilities = clf.predict_proba(x[test_idx])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        rows.append(
            _score_row(labels[test_idx], predictions, probabilities, model_kind, fold, layer, phase)
        )
    return rows


def _cross_validated_text_scores(
    texts: list[str],
    labels: np.ndarray,
    groups: np.ndarray,
    config: ProbeConfig,
    seed: int,
    model_kind: str,
) -> list[dict]:
    rows = []
    text_array = np.array(texts)
    splitter = _splitter(labels, groups, config, seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(text_array, labels, groups)):
        clf = make_pipeline(
            TfidfVectorizer(
                min_df=2,
                ngram_range=(1, 2),
                max_features=20_000,
                token_pattern=r"(?u)\b\w+\b",
            ),
            LogisticRegression(
                C=config.regularization_c,
                class_weight="balanced",
                max_iter=config.max_iter,
                random_state=seed,
            ),
        )
        clf.fit(text_array[train_idx], labels[train_idx])
        probabilities = clf.predict_proba(text_array[test_idx])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        rows.append(
            _score_row(labels[test_idx], predictions, probabilities, model_kind, fold, None, None)
        )
    return rows


def _score_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_kind: str,
    fold: int,
    layer: int | None,
    phase: str | None,
) -> dict:
    return {
        "model_kind": model_kind,
        "fold": fold,
        "layer": layer,
        "phase": phase,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_prob),
        "n_test": len(y_true),
    }
