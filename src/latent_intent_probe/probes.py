from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from latent_intent_probe.config import ProbeConfig


def fit_activation_probes(
    activations_path: str | Path,
    records: list[dict],
    config: ProbeConfig,
    seed: int,
) -> pd.DataFrame:
    activations, phase_names = _load_activations(activations_path)
    binary_idx = _binary_indices(records)
    labels = np.array([records[i]["label"] for i in binary_idx])
    groups = np.array([records[i]["template_id"] for i in binary_idx])
    x_all = activations[binary_idx]

    rows = []
    for phase_idx, phase in enumerate(phase_names):
        for layer in range(x_all.shape[2]):
            scores = _cross_validated_scores(
                x_all[:, phase_idx, layer, :],
                labels,
                groups,
                config,
                seed,
                model_kind="phase_activation",
                layer=layer,
                phase=phase,
            )
            rows.extend(scores)
    return pd.DataFrame(rows)


def fit_phase_transfer_probes(
    activations_path: str | Path,
    records: list[dict],
    activation_metrics: pd.DataFrame,
    config: ProbeConfig,
    seed: int,
) -> pd.DataFrame:
    activations, phase_names = _load_activations(activations_path)
    binary_idx = _binary_indices(records)
    labels = np.array([records[i]["label"] for i in binary_idx])
    groups = np.array([records[i]["template_id"] for i in binary_idx])
    x_all = activations[binary_idx]
    top_layers = _top_layers(activation_metrics, config.transfer_top_k_layers)

    rows = []
    splitter = _splitter(labels, groups, config, seed)
    folds = list(splitter.split(x_all, labels, groups))
    for layer in top_layers:
        for train_phase_idx, train_phase in enumerate(phase_names):
            for test_phase_idx, test_phase in enumerate(phase_names):
                for fold, (train_idx, test_idx) in enumerate(folds):
                    clf = _activation_clf(config, seed)
                    clf.fit(x_all[train_idx, train_phase_idx, layer, :], labels[train_idx])
                    probabilities = clf.predict_proba(x_all[test_idx, test_phase_idx, layer, :])[:, 1]
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
    return pd.DataFrame(rows)


def fit_text_baselines(records: list[dict], config: ProbeConfig, seed: int) -> pd.DataFrame:
    binary_idx = _binary_indices(records)
    labels = np.array([records[i]["label"] for i in binary_idx])
    groups = np.array([records[i]["template_id"] for i in binary_idx])

    baselines = {
        "output_tfidf": [records[i].get("response", "") for i in binary_idx],
        "prompt_tfidf": [records[i].get("rendered_prompt", "") for i in binary_idx],
        "objective_text_tfidf": [records[i].get("private_objective", "") for i in binary_idx],
    }

    rows = []
    for name, texts in baselines.items():
        if not any(text.strip() for text in texts):
            continue
        rows.extend(_cross_validated_text_scores(texts, labels, groups, config, seed, name))
    return pd.DataFrame(rows)


def summarize_attention_asymmetry(attention_path: str | Path) -> pd.DataFrame:
    path = Path(attention_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    attention = pd.read_csv(path)
    attention = attention[attention["label"].isin([0, 1])].copy()
    if attention.empty:
        return pd.DataFrame()

    rows = []
    grouped = attention.groupby(["query_phase", "layer", "head"], dropna=False)
    for (phase, layer, head), group in grouped:
        harmful = group[group["label"] == 1]["goal_anchor_asymmetry"].to_numpy()
        benign = group[group["label"] == 0]["goal_anchor_asymmetry"].to_numpy()
        labels = group["label"].to_numpy()
        scores = group["goal_anchor_asymmetry"].to_numpy()
        if len(harmful) < 2 or len(benign) < 2:
            continue
        pooled = np.sqrt((harmful.var(ddof=1) + benign.var(ddof=1)) / 2)
        effect = 0.0 if pooled == 0 else float((harmful.mean() - benign.mean()) / pooled)
        rows.append(
            {
                "query_phase": phase,
                "layer": int(layer),
                "head": int(head),
                "harmful_mean_asymmetry": float(harmful.mean()),
                "benign_mean_asymmetry": float(benign.mean()),
                "mean_delta_harmful_minus_benign": float(harmful.mean() - benign.mean()),
                "cohens_d": effect,
                "abs_cohens_d": abs(effect),
                "auroc_from_asymmetry": roc_auc_score(labels, scores),
                "n": len(group),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("abs_cohens_d", ascending=False)


def summarize_leakage(records: list[dict]) -> pd.DataFrame:
    terms = [
        "credential",
        "exfiltrat",
        "sabotage",
        "phishing",
        "evasion",
        "fraud",
        "malware",
        "surveillance",
        "tamper",
        "social-engineering",
        "private objective",
        "hidden objective",
        "internal objective",
    ]
    rows = []
    for record in records:
        if record.get("label") not in (0, 1):
            continue
        response = record.get("response", "").lower()
        rows.append(
            {
                "id": record["id"],
                "label": record["label"],
                "label_name": record["label_name"],
                "response_chars": len(record.get("response", "")),
                "leak_term_count": sum(term in response for term in terms),
                "mentions_private_channel": any(
                    term in response
                    for term in ["private", "hidden", "internal", "harness", "experiment"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _load_activations(activations_path: str | Path) -> tuple[np.ndarray, list[str]]:
    payload = np.load(activations_path)
    activations = payload["activations"]
    if activations.ndim == 3:
        return activations[:, None, :, :], ["final_prompt"]
    return activations, [str(phase) for phase in payload["phase_names"].tolist()]


def _binary_indices(records: list[dict]) -> list[int]:
    return [idx for idx, record in enumerate(records) if record.get("label") in (0, 1)]


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
        raise ValueError("Need at least two template groups for cross-validation")
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
        rows.append(_score_row(labels[test_idx], predictions, probabilities, model_kind, fold, layer, phase))
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
            TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=20_000),
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
        rows.append(_score_row(labels[test_idx], predictions, probabilities, model_kind, fold, None, None))
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
        "phase": phase,
        "layer": layer,
        "fold": fold,
        "n_test": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_prob),
    }
