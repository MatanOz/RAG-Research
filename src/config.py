"""Configuration loading for the RAG experiment pipelines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DEFAULT_CONFIG_PATH = Path("configs/p0_baseline.yaml")


class RetrievalParams(BaseModel):
    embedding_model: str
    chunking_method: str = "fixed_char"
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    top_k: int = Field(ge=1)
    embedding_batch_size: int = Field(ge=1)

    model_config = ConfigDict(extra="forbid")


class LLMParams(BaseModel):
    model_name: str
    critic_model_name: Optional[str] = None
    temperature: float
    max_tokens: int = Field(gt=0)

    model_config = ConfigDict(extra="forbid")


class PathsConfig(BaseModel):
    data_dir: str
    chroma_dir: str
    output_dir: str

    model_config = ConfigDict(extra="forbid")


class RunControl(BaseModel):
    paper_ids: Optional[List[int]] = None
    max_papers: Optional[int] = Field(default=None, ge=1)
    question_ids: Optional[List[int]] = None
    max_questions_per_paper: Optional[int] = Field(default=None, ge=1)

    model_config = ConfigDict(extra="forbid")


class LoggingConfig(BaseModel):
    level: str = "INFO"
    progress_every: int = Field(default=10, ge=1)

    model_config = ConfigDict(extra="forbid")


class RunnerConfig(BaseModel):
    project_name: str
    pipeline_version: str
    retrieval_params: RetrievalParams
    llm_params: LLMParams
    paths: PathsConfig
    run_control: RunControl = Field(default_factory=RunControl)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = ConfigDict(extra="forbid")


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> RunnerConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping in: {path}")

    return RunnerConfig.model_validate(raw)


def config_as_dict(config: RunnerConfig) -> Dict[str, Any]:
    return config.model_dump(mode="python")


CONFIG = load_config()
CONFIG_DICT = config_as_dict(CONFIG)
