"""Shared LangGraph state for question-level pipeline execution."""

from __future__ import annotations

from typing import Dict, List, NotRequired, TypedDict


class AgentState(TypedDict):
    question: str
    paper_id: int
    question_id: int
    question_type: NotRequired[str]
    retrieved_chunks: List[Dict[str, object]]
    model_answer: str
    draft_answer: NotRequired[str]
    reasoning: NotRequired[str]
    evidence_quotes: NotRequired[List[str]]
    critique_logic: NotRequired[str]
    is_abstained: NotRequired[bool]
    loop_count: NotRequired[int]
    critic_status: NotRequired[str]
    critic_feedback: NotRequired[str]
    new_search_query: NotRequired[str]
    embedding_tokens: int
    llm_input_tokens: int
    llm_output_tokens: int
