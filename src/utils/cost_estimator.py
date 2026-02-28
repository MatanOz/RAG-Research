"""Cost estimation helpers for retrieval + generation runs."""

from __future__ import annotations

from typing import Dict


EMBEDDING_PRICES_PER_MILLION: Dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}

LLM_INPUT_PRICES_PER_MILLION: Dict[str, float] = {
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.5,
    "gpt-4.1-mini": 0.40,
    "gpt-4.1-nano": 0.10,
}

LLM_OUTPUT_PRICES_PER_MILLION: Dict[str, float] = {
    "gpt-4o-mini": 0.60,
    "gpt-4o": 10.0,
    "gpt-4.1-mini": 1.60,
    "gpt-4.1-nano": 0.40,
}


def estimate_cost_usd(
    embedding_model: str,
    llm_model: str,
    embedding_input_tokens: int,
    llm_input_tokens: int,
    llm_output_tokens: int,
) -> float:
    embedding_cost = (embedding_input_tokens / 1_000_000.0) * EMBEDDING_PRICES_PER_MILLION.get(embedding_model, 0.0)
    llm_input_cost = (llm_input_tokens / 1_000_000.0) * LLM_INPUT_PRICES_PER_MILLION.get(llm_model, 0.0)
    llm_output_cost = (llm_output_tokens / 1_000_000.0) * LLM_OUTPUT_PRICES_PER_MILLION.get(llm_model, 0.0)
    return round(embedding_cost + llm_input_cost + llm_output_cost, 8)


def estimate_llm_only_cost_usd(
    llm_model: str,
    llm_input_tokens: int,
    llm_output_tokens: int,
) -> float:
    llm_input_cost = (llm_input_tokens / 1_000_000.0) * LLM_INPUT_PRICES_PER_MILLION.get(llm_model, 0.0)
    llm_output_cost = (llm_output_tokens / 1_000_000.0) * LLM_OUTPUT_PRICES_PER_MILLION.get(llm_model, 0.0)
    return round(llm_input_cost + llm_output_cost, 8)


def estimate_embedding_only_cost_usd(embedding_model: str, embedding_input_tokens: int) -> float:
    return round(
        (embedding_input_tokens / 1_000_000.0) * EMBEDDING_PRICES_PER_MILLION.get(embedding_model, 0.0),
        8,
    )
