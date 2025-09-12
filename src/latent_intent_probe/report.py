from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from latent_intent_probe.config import ExperimentConfig


def write_report(
    run_dir: str | Path,
    config: ExperimentConfig,
    activation_metrics: pd.DataFrame,
    transfer_metrics: pd.DataFrame,
    text_metrics: pd.DataFrame,
    attention_summary: pd.DataFrame,
    leakage: pd.DataFrame,
) -> Path:
    run_dir = Path(run_dir)
    activation_summary = (
        activation_metrics.groupby(["phase", "layer"], dropna=False)
        .agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"), bal_acc_mean=("balanced_accuracy", "mean"))
        .reset_index()
        .sort_values("auroc_mean", ascending=False)
    )
    phase_summary = (
        activation_metrics.groupby("phase", dropna=False)
        .agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"), bal_acc_mean=("balanced_accuracy", "mean"))
        .reset_index()
        .sort_values("auroc_mean", ascending=False)
    )
    transfer_summary = (
        transfer_metrics.groupby(["train_phase", "test_phase"], dropna=False)
        .agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"), bal_acc_mean=("balanced_accuracy", "mean"))
        .reset_index()
        .sort_values("auroc_mean", ascending=False)
        if not transfer_metrics.empty
        else pd.DataFrame()
    )
    text_summary = (
        text_metrics.groupby(["model_kind"], dropna=False)
        .agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"), bal_acc_mean=("balanced_accuracy", "mean"))
        .reset_index()
        .sort_values("auroc_mean", ascending=False)
        if not text_metrics.empty
        else pd.DataFrame()
    )

    activation_summary.to_csv(run_dir / "activation_probe_summary.csv", index=False)
    phase_summary.to_csv(run_dir / "phase_probe_summary.csv", index=False)
    transfer_summary.to_csv(run_dir / "phase_transfer_summary.csv", index=False)
    text_summary.to_csv(run_dir / "text_baseline_summary.csv", index=False)
    attention_summary.to_csv(run_dir / "attention_asymmetry_summary.csv", index=False)
    leakage.to_csv(run_dir / "output_leakage.csv", index=False)

    _plot_phase_layer_auroc(activation_metrics, run_dir / "phase_layer_auroc.png")
    if not transfer_summary.empty:
        _plot_transfer_heatmap(transfer_summary, run_dir / "phase_transfer_auroc.png")

    top_phase_layers = activation_summary.head(config.report.top_k_layers)
    top_heads = attention_summary.head(config.report.top_k_heads) if not attention_summary.empty else pd.DataFrame()
    leak_rate = float((leakage["leak_term_count"] > 0).mean()) if not leakage.empty else 0.0
    private_rate = float(leakage["mentions_private_channel"].mean()) if not leakage.empty else 0.0

    report_path = run_dir / "summary.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Temporal Latent Intent Probe Run: {config.run.name}\n\n")
        handle.write("## Setup\n\n")
        handle.write(f"- Model: `{config.model.name}`\n")
        handle.write(f"- Binary examples per label: `{config.dataset.examples_per_label}`\n")
        handle.write("- Activation readouts: objective span, post-objective token, public-task span, final prompt token, first response token, response mean\n")
        handle.write("- Attention readout: per-head goal-anchor asymmetry, `attention_density_to_objective - attention_density_to_public_task`\n")
        handle.write("- Split: stratified group CV by prompt template\n\n")

        handle.write("## Phase Summary\n\n")
        handle.write(phase_summary.to_markdown(index=False))
        handle.write("\n\n")

        handle.write("## Best Phase-Layer Activation Readouts\n\n")
        handle.write(top_phase_layers.to_markdown(index=False))
        handle.write("\n\n")

        handle.write("## Phase Transfer\n\n")
        if transfer_summary.empty:
            handle.write("No phase-transfer rows were produced.\n\n")
        else:
            handle.write(transfer_summary.head(20).to_markdown(index=False))
            handle.write("\n\n")

        handle.write("## Goal-Anchor Attention Asymmetry\n\n")
        if top_heads.empty:
            handle.write("No attention rows were produced. Check whether the model supports `output_attentions=True` with the configured attention implementation.\n\n")
        else:
            handle.write(top_heads.to_markdown(index=False))
            handle.write("\n\n")

        handle.write("## Text Baselines\n\n")
        if text_summary.empty:
            handle.write("No text baseline rows were produced.\n\n")
        else:
            handle.write(text_summary.to_markdown(index=False))
            handle.write("\n\n")

        handle.write("## Output Leakage Check\n\n")
        handle.write(f"- Responses with harmful/private lexicon hits: `{leak_rate:.3f}`\n")
        handle.write(f"- Responses mentioning private/internal channel: `{private_rate:.3f}`\n\n")
        handle.write("## Files\n\n")
        handle.write("- `records_with_responses.jsonl`: prompts, phase token counts, labels, and generated text\n")
        handle.write("- `activations_by_phase.npz`: activation tensor `[example, phase, layer, hidden]`\n")
        handle.write("- `attention_asymmetry_rows.csv`: per-example, per-layer, per-head objective/task attention mass\n")
        handle.write("- `activation_probe_metrics.csv`: per-fold phase/layer probe metrics\n")
        handle.write("- `phase_transfer_metrics.csv`: train-phase to test-phase generalization metrics\n")
        handle.write("- `attention_asymmetry_summary.csv`: head rankings by harmful-vs-benign asymmetry effect\n")
        handle.write("- `phase_layer_auroc.png`: phase-by-layer activation AUROC plot\n")
        handle.write("- `phase_transfer_auroc.png`: train/test phase transfer heatmap\n\n")
        handle.write("## Interpretation Guardrails\n\n")
        handle.write(
            "This run asks when the private-objective signal is linearly readable and whether "
            "a concrete computation pattern, goal-anchor attention asymmetry, covaries with that "
            "signal. It still does not prove autonomous intent: the objective is prompt-injected. "
            "The useful evidence is temporal localization, phase transfer or non-transfer, and "
            "whether particular heads route attention differently to objective versus public-task spans.\n"
        )
    return report_path


def _plot_phase_layer_auroc(metrics: pd.DataFrame, path: Path) -> None:
    pivot = metrics.groupby(["phase", "layer"], dropna=False)["auroc"].mean().reset_index()
    table = pivot.pivot(index="phase", columns="layer", values="auroc")
    plt.figure(figsize=(12, 5))
    sns.heatmap(table, vmin=0.0, vmax=1.0, cmap="viridis", cbar_kws={"label": "AUROC"})
    plt.title("Private-objective probe AUROC by phase and layer")
    plt.xlabel("Layer")
    plt.ylabel("Phase")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_transfer_heatmap(summary: pd.DataFrame, path: Path) -> None:
    table = summary.pivot(index="train_phase", columns="test_phase", values="auroc_mean")
    plt.figure(figsize=(8, 6))
    sns.heatmap(table, vmin=0.0, vmax=1.0, cmap="mako", annot=True, fmt=".2f", cbar_kws={"label": "AUROC"})
    plt.title("Phase-transfer AUROC")
    plt.xlabel("Test phase")
    plt.ylabel("Train phase")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
