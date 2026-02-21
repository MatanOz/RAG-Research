"""Shared LangGraph state for question-level pipeline execution."""

from __future__ import annotations

from typing import Dict, List, TypedDict


class AgentState(TypedDict):
    question: str
    paper_id: int
    question_id: int
    retrieved_chunks: List[Dict[str, object]]
    model_answer: str
    embedding_tokens: int
    llm_input_tokens: int
    llm_output_tokens: int
