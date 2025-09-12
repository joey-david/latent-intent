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
    text_metrics: pd.DataFrame,
    leakage: pd.DataFrame,
) -> Path:
    run_dir = Path(run_dir)
    activation_summary = (
        activation_metrics.groupby(["model_kind", "layer"], dropna=False)
        .agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"), bal_acc_mean=("balanced_accuracy", "mean"))
        .reset_index()
        .sort_values("auroc_mean", ascending=False)
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
    text_summary.to_csv(run_dir / "text_baseline_summary.csv", index=False)
    leakage.to_csv(run_dir / "output_leakage.csv", index=False)

    plot_path = run_dir / "layer_auroc.png"
    _plot_layer_auroc(activation_metrics, plot_path)

    top = activation_summary.head(config.report.top_k_layers)
    leak_rate = float((leakage["leak_term_count"] > 0).mean()) if not leakage.empty else 0.0
    private_rate = float(leakage["mentions_private_channel"].mean()) if not leakage.empty else 0.0

    report_path = run_dir / "summary.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Latent Intent Probe Run: {config.run.name}\n\n")
        handle.write("## Setup\n\n")
        handle.write(f"- Model: `{config.model.name}`\n")
        handle.write(f"- Binary examples per label: `{config.dataset.examples_per_label}`\n")
        handle.write("- Activation readout: final prompt token, all transformer layers\n")
        handle.write("- Split: stratified group CV by prompt template\n\n")

        handle.write("## Best Activation Layers\n\n")
        handle.write(top.to_markdown(index=False))
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
        handle.write("- `records_with_responses.jsonl`: prompts, labels, and generated text\n")
        handle.write("- `activations_last_prompt_token.npz`: activation tensor `[example, layer, hidden]`\n")
        handle.write("- `activation_probe_metrics.csv`: per-fold activation probe metrics\n")
        handle.write("- `text_baseline_metrics.csv`: per-fold output/prompt TF-IDF baselines\n")
        handle.write("- `layer_auroc.png`: layer-wise activation AUROC plot\n\n")
        handle.write("## Interpretation Guardrails\n\n")
        handle.write(
            "This run tests whether a private objective signal remains linearly decodable from "
            "hidden states when the generated answer is constrained to be neutral. It does not, by "
            "itself, prove that a model has autonomous intent. Prompt and output baselines are "
            "included to separate activation readout from obvious surface leakage.\n"
        )
    return report_path


def _plot_layer_auroc(metrics: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=metrics, x="layer", y="auroc", marker="o", errorbar="sd")
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylim(0.0, 1.02)
    plt.title("Private-objective probe AUROC by layer")
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
