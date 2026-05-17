from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv() -> bool:
        return False

from latent_intent_probe.config import load_config, write_resolved_config
from latent_intent_probe.dataset import build_dataset, records_to_dicts, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run counterfactual selected-objective probing experiments."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the paired dataset/run directory without loading a model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional record cap for quick server smoke tests; complete pairs are preserved.",
    )
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    run_dir = _make_run_dir(config.run.output_dir, config.run.name)
    write_resolved_config(config, run_dir / "resolved_config.yaml")

    records = records_to_dicts(build_dataset(config.dataset, config.run.seed))
    if args.limit is not None:
        records = _balanced_limit(records, args.limit)
    _validate_counterfactual_records(records)
    write_jsonl(records, run_dir / "dataset.jsonl")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "n_records": len(records),
        "n_pairs": len({record["pair_id"] for record in records}),
        "fixed_response": config.dataset.fixed_response,
        "heldout_group": "scenario_id",
        "pair_invariants_validated": True,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.dry_run:
        print(f"Dry run complete. Wrote paired dataset/config to {run_dir}")
        print("Pair invariants validated: only the A/B selector changes within each pair.")
        return

    from latent_intent_probe.hf_inference import (
        collect_phase_readouts_and_generations,
        load_model_and_tokenizer,
    )
    from latent_intent_probe.probes import (
        fit_activation_probes,
        fit_phase_transfer_probes,
        fit_text_baselines,
        summarize_attention_asymmetry,
        summarize_output_control,
        summarize_paired_directions,
    )
    from latent_intent_probe.report import write_report

    model, tokenizer = load_model_and_tokenizer(config.model)
    activations_path, records_path, attention_path = collect_phase_readouts_and_generations(
        records,
        model,
        tokenizer,
        config.model,
        run_dir,
    )

    enriched_records = _read_jsonl(records_path)
    activation_metrics = fit_activation_probes(
        activations_path,
        enriched_records,
        config.probe,
        config.run.seed,
    )
    transfer_metrics = fit_phase_transfer_probes(
        activations_path,
        enriched_records,
        activation_metrics,
        config.probe,
        config.run.seed,
    )
    text_metrics = fit_text_baselines(enriched_records, config.probe, config.run.seed)
    paired_directions = summarize_paired_directions(
        activations_path,
        enriched_records,
        config.probe,
        config.run.seed,
    )
    attention_summary = summarize_attention_asymmetry(attention_path)
    output_control = summarize_output_control(enriched_records)

    activation_metrics.to_csv(run_dir / "activation_probe_metrics.csv", index=False)
    transfer_metrics.to_csv(run_dir / "phase_transfer_metrics.csv", index=False)
    text_metrics.to_csv(run_dir / "text_baseline_metrics.csv", index=False)
    paired_directions.to_csv(run_dir / "paired_direction_summary.csv", index=False)
    attention_summary.to_csv(run_dir / "attention_asymmetry_summary.csv", index=False)
    output_control.to_csv(run_dir / "output_control.csv", index=False)

    report_path = write_report(
        run_dir,
        config,
        activation_metrics,
        transfer_metrics,
        text_metrics,
        paired_directions,
        attention_summary,
        output_control,
    )
    print(f"Experiment complete. Report: {report_path}")
    print(f"Hero figure: {run_dir / 'counterfactual_result.png'}")


def _make_run_dir(output_dir: str, run_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / f"{timestamp}-{_slugify(run_name)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _slugify(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")


def _balanced_limit(records: list[dict], limit: int) -> list[dict]:
    """Limit a smoke run without breaking matched counterfactual pairs."""
    if limit >= len(records):
        return records
    if limit < 2:
        return records[:limit]

    pair_order = []
    seen = set()
    for record in records:
        pair_id = record["pair_id"]
        if pair_id not in seen:
            seen.add(pair_id)
            pair_order.append(pair_id)

    pair_count = max(1, limit // 2)
    selected = set(pair_order[:pair_count])
    limited = [record for record in records if record["pair_id"] in selected]
    return limited[: 2 * pair_count]


def _validate_counterfactual_records(records: list[dict]) -> None:
    by_pair: dict[str, list[dict]] = {}
    for record in records:
        by_pair.setdefault(str(record["pair_id"]), []).append(record)

    for pair_id, pair in by_pair.items():
        if len(pair) != 2:
            raise ValueError(f"{pair_id} is incomplete: expected 2 records, got {len(pair)}")
        if {int(record["label"]) for record in pair} != {0, 1}:
            raise ValueError(f"{pair_id} does not contain one benign-active and one harmful-active row")
        if len({record["fixed_response"] for record in pair}) != 1:
            raise ValueError(f"{pair_id} does not share one fixed public response")

        normalized_prompts = []
        for record in pair:
            prompt = record["messages"][1]["content"]
            selector = f"ACTIVE OBJECTIVE: {record['active_slot']}"
            normalized_prompts.append(prompt.replace(selector, "ACTIVE OBJECTIVE: X", 1))
        if normalized_prompts[0] != normalized_prompts[1]:
            raise ValueError(f"{pair_id} differs in text outside the active A/B selector")


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
