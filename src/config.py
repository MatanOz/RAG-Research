"""YAML-based configuration loader for experiment settings."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DEFAULT_CONFIG_PATH = Path("configs/p0_baseline.yaml")


@dataclass(frozen=True)
class RetrievalParams:
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


@dataclass(frozen=True)
class LLMParams:
    model_name: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class Paths:
    data_dir: str
    db_dir: str
    output_dir: str


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    pipeline_version: str
    retrieval_params: RetrievalParams
    llm_params: LLMParams
    paths: Paths


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and parse a YAML experiment config into a typed object."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping in: {path}")

    try:
        return AppConfig(
            project_name=raw["project_name"],
            pipeline_version=raw["pipeline_version"],
            retrieval_params=RetrievalParams(**raw["retrieval_params"]),
            llm_params=LLMParams(**raw["llm_params"]),
            paths=Paths(**raw["paths"]),
        )
    except KeyError as exc:
        missing_key = exc.args[0]
        raise ValueError(f"Missing required config key: {missing_key}") from exc


def config_as_dict(config: AppConfig) -> dict[str, Any]:
    """Return a plain dictionary representation of an AppConfig object."""
    return asdict(config)


CONFIG = load_config()
CONFIG_DICT = config_as_dict(CONFIG)
