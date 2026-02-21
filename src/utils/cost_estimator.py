"""Cost estimation helpers for retrieval + generation runs."""

from __future__ import annotations


def estimate_cost_usd(
    embedding_model: str,
    llm_model: str,
    embedding_input_tokens: int,
    llm_input_tokens: int,
    llm_output_tokens: int,
) -> float:
    embedding_prices = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
    }
    llm_input_prices = {
        "gpt-4o-mini": 0.15,
        "gpt-4.1-mini": 0.40,
        "gpt-4.1-nano": 0.10,
    }
    llm_output_prices = {
        "gpt-4o-mini": 0.60,
        "gpt-4.1-mini": 1.60,
        "gpt-4.1-nano": 0.40,
    }

    embedding_cost = (embedding_input_tokens / 1_000_000.0) * embedding_prices.get(embedding_model, 0.0)
    llm_input_cost = (llm_input_tokens / 1_000_000.0) * llm_input_prices.get(llm_model, 0.0)
    llm_output_cost = (llm_output_tokens / 1_000_000.0) * llm_output_prices.get(llm_model, 0.0)
    return round(embedding_cost + llm_input_cost + llm_output_cost, 8)


def estimate_embedding_only_cost_usd(embedding_model: str, embedding_input_tokens: int) -> float:
    embedding_prices = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
    }
    return round(
        (embedding_input_tokens / 1_000_000.0) * embedding_prices.get(embedding_model, 0.0),
        8,
    )
