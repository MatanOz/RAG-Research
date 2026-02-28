"""Cost and efficiency helpers for offline evaluation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import tiktoken


def count_text_tokens(text: str, model_name: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text or ""))


def _embedding_price_per_million(pricing: Mapping[str, Any], embedding_model: str) -> float:
    embedding_prices = pricing.get("embedding", {}) if isinstance(pricing, dict) else {}
    return float(embedding_prices.get(embedding_model, 0.0))


def _llm_prices(pricing: Mapping[str, Any], llm_model: str) -> Tuple[float, float]:
    llm_pricing = pricing.get("llm", {}) if isinstance(pricing, dict) else {}
    model_pricing = llm_pricing.get(llm_model, {}) if isinstance(llm_pricing, dict) else {}
    input_price = float(model_pricing.get("input_per_million", 0.0))
    output_price = float(model_pricing.get("output_per_million", 0.0))
    return input_price, output_price


def estimate_question_cost(record: Mapping[str, Any], pricing: Mapping[str, Any]) -> Dict[str, float]:
    models = record.get("models", {}) if isinstance(record, dict) else {}
    logs = record.get("logs", {}) if isinstance(record, dict) else {}

    embedding_model = str(models.get("embedding_model", "text-embedding-3-small"))
    llm_model = str(models.get("llm_model", "gpt-4o-mini"))
    question = str(record.get("question", ""))

    query_embedding_tokens = count_text_tokens(question, embedding_model)
    query_embedding_price = _embedding_price_per_million(pricing, embedding_model)
    query_embedding_cost_usd = (query_embedding_tokens / 1_000_000.0) * query_embedding_price

    tokens_input = int(logs.get("tokens_input", 0))
    tokens_output = int(logs.get("tokens_output", 0))
    llm_input_tokens_est = max(tokens_input - query_embedding_tokens, 0)

    drafter_model = str(logs.get("drafter_model", "") or llm_model)
    critic_model = str(logs.get("critic_model", "") or "")
    drafter_tokens_input = int(logs.get("drafter_tokens_input", 0) or 0)
    drafter_tokens_output = int(logs.get("drafter_tokens_output", 0) or 0)
    critic_tokens_input = int(logs.get("critic_tokens_input", 0) or 0)
    critic_tokens_output = int(logs.get("critic_tokens_output", 0) or 0)

    has_split_llm_usage = any(
        value > 0
        for value in [
            drafter_tokens_input,
            drafter_tokens_output,
            critic_tokens_input,
            critic_tokens_output,
        ]
    )

    if has_split_llm_usage:
        drafter_input_price, drafter_output_price = _llm_prices(pricing, drafter_model)
        critic_input_price, critic_output_price = _llm_prices(pricing, critic_model)
        generation_cost_usd = (
            (drafter_tokens_input / 1_000_000.0) * drafter_input_price
            + (drafter_tokens_output / 1_000_000.0) * drafter_output_price
            + (critic_tokens_input / 1_000_000.0) * critic_input_price
            + (critic_tokens_output / 1_000_000.0) * critic_output_price
        )
    else:
        llm_input_price, llm_output_price = _llm_prices(pricing, llm_model)
        generation_cost_usd = (
            (llm_input_tokens_est / 1_000_000.0) * llm_input_price
            + (tokens_output / 1_000_000.0) * llm_output_price
        )

    total_eval_cost_usd = query_embedding_cost_usd + generation_cost_usd
    if total_eval_cost_usd <= 0:
        total_eval_cost_usd = float(logs.get("estimated_cost_usd", 0.0))

    return {
        "query_embedding_tokens": float(query_embedding_tokens),
        "query_embedding_cost_usd": float(query_embedding_cost_usd),
        "generation_cost_usd": float(generation_cost_usd),
        "total_eval_cost_usd": float(total_eval_cost_usd),
    }
