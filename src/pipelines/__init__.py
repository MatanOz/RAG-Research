"""Pipeline registry and exports."""

from __future__ import annotations

from src.pipelines.p0_pipeline import P0_Pipeline
from src.pipelines.p1_pipeline import P1_Pipeline
from src.pipelines.p2_pipeline import P2_Pipeline

PIPELINE_REGISTRY = {
    "P0": P0_Pipeline,
    "P1": P1_Pipeline,
    "P2": P2_Pipeline,
}

__all__ = [
    "P0_Pipeline",
    "P1_Pipeline",
    "P2_Pipeline",
    "PIPELINE_REGISTRY",
]
