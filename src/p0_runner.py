"""Backward-compatible entrypoint for the graph-based P0 runner."""

from __future__ import annotations

from pathlib import Path

from src.main import DEFAULT_GOLD_PATH, parse_args, run_pipeline


def run_p0_baseline(
    config_path: Path = Path("configs/p0_baseline.yaml"),
    gold_path: Path = DEFAULT_GOLD_PATH,
) -> Path:
    return run_pipeline(config_path=config_path, gold_path=gold_path)


if __name__ == "__main__":
    args = parse_args()
    path = run_pipeline(config_path=args.config, gold_path=args.gold)
    print(f"Wrote baseline run to: {path}")
