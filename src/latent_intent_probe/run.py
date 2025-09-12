from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - lets local dry-runs work before install
    def load_dotenv() -> bool:
        return False

from latent_intent_probe.config import load_config, write_resolved_config
from latent_intent_probe.dataset import build_dataset, records_to_dicts, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run latent private-objective probing experiments.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to experiment YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Build dataset and run directory without loading a model.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on records for quick server smoke tests.")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    run_dir = _make_run_dir(config.run.output_dir, config.run.name)
    write_resolved_config(config, run_dir / "resolved_config.yaml")

    records = records_to_dicts(build_dataset(config.dataset, config.run.seed))
    if args.limit is not None:
        records = _balanced_limit(records, args.limit)
    write_jsonl(records, run_dir / "dataset.jsonl")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "n_records": len(records),
        "n_binary_records": sum(record["label"] in (0, 1) for record in records),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(f"Dry run complete. Wrote dataset/config to {run_dir}")
        print("Run without --dry-run on the SSH server to download the model and start inference.")
        return

    from latent_intent_probe.hf_inference import collect_phase_readouts_and_generations, load_model_and_tokenizer
    from latent_intent_probe.probes import (
        fit_activation_probes,
        fit_phase_transfer_probes,
        fit_text_baselines,
        summarize_attention_asymmetry,
        summarize_leakage,
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
    activation_metrics = fit_activation_probes(activations_path, enriched_records, config.probe, config.run.seed)
    transfer_metrics = fit_phase_transfer_probes(
        activations_path,
        enriched_records,
        activation_metrics,
        config.probe,
        config.run.seed,
    )
    text_metrics = fit_text_baselines(enriched_records, config.probe, config.run.seed)
    attention_summary = summarize_attention_asymmetry(attention_path)
    leakage = summarize_leakage(enriched_records)

    activation_metrics.to_csv(run_dir / "activation_probe_metrics.csv", index=False)
    transfer_metrics.to_csv(run_dir / "phase_transfer_metrics.csv", index=False)
    text_metrics.to_csv(run_dir / "text_baseline_metrics.csv", index=False)
    attention_summary.to_csv(run_dir / "attention_asymmetry_summary.csv", index=False)
    leakage.to_csv(run_dir / "output_leakage_rows.csv", index=False)

    report_path = write_report(
        run_dir,
        config,
        activation_metrics,
        transfer_metrics,
        text_metrics,
        attention_summary,
        leakage,
    )
    print(f"Experiment complete. Report: {report_path}")


def _make_run_dir(output_dir: str, run_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / f"{timestamp}-{_slugify(run_name)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _slugify(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")


def _balanced_limit(records: list[dict], limit: int) -> list[dict]:
    if limit >= len(records):
        return records
    if limit < 4:
        return records[:limit]

    harmful = [record for record in records if record["label"] == 1]
    benign = [record for record in records if record["label"] == 0]
    neutral = [record for record in records if record["label"] == 2]
    per_label = max(2, limit // 2)
    selected = harmful[:per_label] + benign[:per_label]
    remaining = limit - len(selected)
    if remaining > 0:
        selected.extend(neutral[:remaining])
    return selected[:limit]


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
