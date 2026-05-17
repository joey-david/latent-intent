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
    paired_directions: pd.DataFrame,
    attention_summary: pd.DataFrame,
    output_control: pd.DataFrame,
) -> Path:
    run_dir = Path(run_dir)
    activation_summary = (
        activation_metrics.groupby(["phase", "layer"], dropna=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            bal_acc_mean=("balanced_accuracy", "mean"),
        )
        .reset_index()
        .sort_values("auroc_mean", ascending=False)
    )
    phase_summary = (
        activation_metrics.groupby("phase", dropna=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            bal_acc_mean=("balanced_accuracy", "mean"),
        )
        .reset_index()
    )
    transfer_summary = (
        transfer_metrics.groupby(["train_phase", "test_phase"], dropna=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            bal_acc_mean=("balanced_accuracy", "mean"),
        )
        .reset_index()
        if not transfer_metrics.empty
        else pd.DataFrame()
    )
    text_summary = (
        text_metrics.groupby("model_kind", dropna=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            bal_acc_mean=("balanced_accuracy", "mean"),
        )
        .reset_index()
        if not text_metrics.empty
        else pd.DataFrame()
    )

    activation_summary.to_csv(run_dir / "activation_probe_summary.csv", index=False)
    phase_summary.to_csv(run_dir / "phase_probe_summary.csv", index=False)
    transfer_summary.to_csv(run_dir / "phase_transfer_summary.csv", index=False)
    text_summary.to_csv(run_dir / "text_baseline_summary.csv", index=False)

    _plot_counterfactual_result(
        activation_summary,
        paired_directions,
        run_dir / "counterfactual_result.png",
    )
    if not transfer_summary.empty:
        _plot_transfer_heatmap(transfer_summary, run_dir / "phase_transfer_auroc.png")

    match_rate = (
        float(output_control["matches_fixed_response"].mean())
        if not output_control.empty
        else 0.0
    )
    top_heads = (
        attention_summary[
            attention_summary["query_phase"].isin(
                ["post_selector", "public_task_span", "response_first", "response_mean"]
            )
        ].head(config.report.top_k_heads)
        if not attention_summary.empty
        else pd.DataFrame()
    )

    report_path = run_dir / "summary.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Counterfactual Latent Intent: {config.run.name}\n\n")
        handle.write(
            "Two prompt conditions contain the same objective pair, public task, and "
            "teacher-forced response. They differ only in whether objective A or B is marked active. "
            "Which slot contains the harmful objective is randomized across pairs.\n\n"
        )

        handle.write("## Controls\n\n")
        handle.write(f"- Model: `{config.model.name}`\n")
        handle.write(f"- Matched counterfactual pairs: `{len(output_control) // 2}`\n")
        handle.write("- Cross-validation group: held-out `scenario_id` objective domains\n")
        handle.write(f"- Exact fixed-output match rate: `{match_rate:.3f}`\n")
        handle.write(
            "- `pre_selector` is a negative control: pair members are byte-identical up to "
            "the A/B selector.\n\n"
        )

        handle.write("## Probe AUROC By Phase\n\n")
        handle.write(phase_summary.to_markdown(index=False))
        handle.write("\n\n")

        handle.write("## Best Phase-Layer Readouts\n\n")
        handle.write(activation_summary.head(config.report.top_k_layers).to_markdown(index=False))
        handle.write("\n\n")

        handle.write("## Lexical Controls\n\n")
        if text_summary.empty:
            handle.write("No text baseline rows were produced.\n\n")
        else:
            handle.write(text_summary.to_markdown(index=False))
            handle.write("\n\n")

        handle.write("## Shared Counterfactual Directions\n\n")
        if paired_directions.empty:
            handle.write("No complete counterfactual pairs were available.\n\n")
        else:
            strongest = paired_directions.sort_values(
                ["heldout_cosine_mean", "direction_coherence"],
                ascending=False,
            ).head(config.report.top_k_layers)
            handle.write(strongest.to_markdown(index=False))
            handle.write("\n\n")
            handle.write(
                "`heldout_cosine_mean` evaluates a mean harmful-minus-benign direction on "
                "objective domains excluded from that direction's construction. `permutation_p` "
                "uses random sign flips of paired deltas.\n\n"
            )

        handle.write("## Phase Transfer\n\n")
        if transfer_summary.empty:
            handle.write("No phase-transfer rows were produced.\n\n")
        else:
            handle.write(
                transfer_summary.sort_values("auroc_mean", ascending=False)
                .head(20)
                .to_markdown(index=False)
            )
            handle.write("\n\n")

        handle.write("## Active-vs-Inactive Objective Attention\n\n")
        if top_heads.empty:
            handle.write(
                "No attention rows were produced. Check whether the configured model supports "
                "`output_attentions=True` with eager attention.\n\n"
            )
        else:
            handle.write(top_heads.to_markdown(index=False))
            handle.write("\n\n")

        handle.write("## Main Artifacts\n\n")
        handle.write("- `counterfactual_result.png`: probe AUROC and paired-direction heatmaps\n")
        handle.write("- `phase_probe_summary.csv`: phase-level activation probe results\n")
        handle.write("- `paired_direction_summary.csv`: counterfactual direction statistics\n")
        handle.write("- `phase_transfer_summary.csv`: train-phase to test-phase generalization\n")
        handle.write("- `attention_asymmetry_summary.csv`: active-vs-inactive objective attention heads\n")
        handle.write("- `text_baseline_summary.csv`: lexical negative controls\n")
        handle.write("- `output_control.csv`: verifies the public response is identical\n\n")

        handle.write("## Interpretation\n\n")
        handle.write(
            "This experiment studies a synthetic, prompt-injected selected objective, not "
            "autonomous malicious intent. The interesting evidence is whether a label that is "
            "undefined before the selector becomes readable afterward, generalizes to held-out "
            "objective domains, persists into an identical response, and yields a shared paired "
            "activation direction or active-objective attention preference.\n"
        )
    return report_path


def _plot_counterfactual_result(
    activation_summary: pd.DataFrame,
    paired_directions: pd.DataFrame,
    path: Path,
) -> None:
    phase_order = [
        "pre_selector",
        "post_selector",
        "public_task_span",
        "final_prompt",
        "response_first",
        "response_mean",
    ]
    probe_table = activation_summary.pivot(
        index="phase",
        columns="layer",
        values="auroc_mean",
    ).reindex(phase_order)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    sns.heatmap(
        probe_table,
        vmin=0.4,
        vmax=1.0,
        cmap="viridis",
        ax=axes[0],
        cbar_kws={"label": "AUROC"},
    )
    axes[0].set_title("Held-out-domain probe: which private objective is active?")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Phase")

    if paired_directions.empty:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No paired-direction data", ha="center", va="center")
    else:
        direction_table = paired_directions.pivot(
            index="phase",
            columns="layer",
            values="heldout_cosine_mean",
        ).reindex(phase_order)
        sns.heatmap(
            direction_table,
            center=0.0,
            cmap="vlag",
            ax=axes[1],
            cbar_kws={"label": "held-out cosine"},
        )
        axes[1].set_title("Shared harmful-minus-benign counterfactual direction")
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Phase")

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_transfer_heatmap(summary: pd.DataFrame, path: Path) -> None:
    table = summary.pivot(index="train_phase", columns="test_phase", values="auroc_mean")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        table,
        vmin=0.0,
        vmax=1.0,
        cmap="mako",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "AUROC"},
    )
    plt.title("Phase-transfer AUROC")
    plt.xlabel("Test phase")
    plt.ylabel("Train phase")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
