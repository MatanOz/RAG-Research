"""Prompt templates used by the semantic evaluation judge."""

from __future__ import annotations


BASE_PROMPT = """
You are an impartial chemistry QA evaluator.
Return a JSON object with exact keys:
- semantic_correctness: number in [0, 1]
- groundedness: number in [0, 1]
- explanation: short string

Scoring rules:
1) semantic_correctness compares model_answer against gold_answer.
2) groundedness checks if model_answer is supported by retrieved_context only.
3) Penalize hallucinations that are not grounded in retrieved_context.
4) Use strict scoring for factual chemistry content.
""".strip()


QUESTION_TYPE_HINTS = {
    "NUMERIC": "For NUMERIC questions, prioritize value and unit correctness.",
    "LIST": "For LIST questions, prioritize completeness and precision of listed items.",
    "STRING": "For STRING questions, prioritize exact factual match and specificity.",
    "CATEGORICAL": "For CATEGORICAL questions, prioritize exact category match.",
    "FREE_TEXT": "For FREE_TEXT questions, prioritize factual consistency and key points coverage.",
}


def get_system_prompt(question_type: str) -> str:
    hint = QUESTION_TYPE_HINTS.get((question_type or "").upper(), QUESTION_TYPE_HINTS["FREE_TEXT"])
    return f"{BASE_PROMPT}\n\nType hint: {hint}"

