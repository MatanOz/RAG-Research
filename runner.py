"""Convenience CLI for running a selected pipeline config."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.main import DEFAULT_GOLD_PATH, run_pipeline

PIPELINE_CONFIGS = {
    "P0": Path("configs/p0_baseline.yaml"),
    "P1": Path("configs/p1_semantic.yaml"),
    "P2": Path("configs/p2_hybrid.yaml"),
    "P2_IMP": Path("configs/p2_imp_hybrid.yaml"),
    "P3": Path("configs/p3_adaptive_structured.yaml"),
    "P4": Path("configs/p4_agentic_corrector.yaml"),
    "P5_VER1": Path("configs/p5_ver1.yaml"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a selected RAG pipeline quickly")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="P0",
        help="Pipeline version to run (P0, P1, P2, P2_IMP, P3, P4, or P5_VER1). Ignored when --config is provided.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional explicit config path.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH, help="Gold JSON path.")
    return parser.parse_args()


def resolve_config_path(pipeline: str, config: Path | None) -> Path:
    if config is not None:
        return config

    pipeline_key = pipeline.upper()
    config_path = PIPELINE_CONFIGS.get(pipeline_key)
    if config_path is None:
        raise ValueError(
            f"Unsupported pipeline '{pipeline}'. Choose one of: {sorted(PIPELINE_CONFIGS.keys())}"
        )
    return config_path


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.pipeline, args.config)
    output_path = run_pipeline(config_path=config_path, gold_path=args.gold)
    print(f"Run complete: {output_path}")


if __name__ == "__main__":
    main()
