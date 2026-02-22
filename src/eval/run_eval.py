"""CLI entrypoint for standalone evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import re
import sys
from pathlib import Path

from src.eval.evaluator import Evaluator
from src.eval.ui_exporter import export_ui_dashboard_data


DEFAULT_CONFIG_PATH = Path("src/eval/config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pipeline JSONL outputs against gold data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Evaluation YAML config path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path override. If a directory is provided, an auto-named file will be created inside it.",
    )
    return parser.parse_args()


def _sanitize_pipeline_label(label: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    return sanitized or "pipeline"


def _build_auto_output_path(base_output_path: Path, pipeline_labels: list[str]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    labels = "_".join(_sanitize_pipeline_label(label) for label in pipeline_labels) or "no_pipelines"
    filename = f"ui_dashboard_data_{timestamp}_{labels}.json"
    return base_output_path / filename


def _resolve_output_path(
    output_arg: Path | None,
    default_output: Path,
    pipeline_labels: list[str],
) -> Path:
    if output_arg is not None:
        if output_arg.suffix.lower() == ".json":
            return output_arg
        return _build_auto_output_path(output_arg, pipeline_labels)

    if default_output.suffix.lower() == ".json":
        return _build_auto_output_path(default_output.parent, pipeline_labels)
    return _build_auto_output_path(default_output, pipeline_labels)


def main() -> int:
    args = parse_args()
    try:
        evaluator = Evaluator(config_path=args.config)
        payload = evaluator.evaluate()
        default_output = Path(evaluator.config.get("paths", {}).get("output_path", "outputs/eval/ui_dashboard_data.json"))
        pipeline_labels = list(payload.get("summary_metrics", {}).keys())
        output_path = _resolve_output_path(
            output_arg=args.output,
            default_output=default_output,
            pipeline_labels=pipeline_labels,
        )
        written = export_ui_dashboard_data(payload=payload, output_path=output_path)
        print(f"Evaluation complete. Wrote dashboard payload to: {written}")
        return 0
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
