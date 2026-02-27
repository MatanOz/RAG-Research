"""P5_ver1 implementation with autonomous critic-driven revise/re-retrieve loops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai import APIError, APIStatusError, AuthenticationError, PermissionDeniedError, RateLimitError
from pydantic import BaseModel, Field

from src.pipelines.p3_pipeline import P3_Pipeline
from src.state import AgentState


def _load_critic_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    normalized: Dict[str, str] = {}
    for key, value in payload.items():
        key_str = str(key).strip()
        value_str = str(value).strip()
        if key_str and value_str:
            normalized[key_str] = value_str
    return normalized


CRITIC_MAP_PATH = Path("specs/critic_instructions_map.json")
CRITIC_MAP = _load_critic_map(CRITIC_MAP_PATH)


class P5DraftResponse(BaseModel):
    draft_answer: str
    reasoning: str
    quotes: List[str]


class P5CriticResponse(BaseModel):
    status: Literal["ACCEPT", "REVISE", "RE_RETRIEVE", "ABSTAIN"] = Field(
        description="Decision on the next step."
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Feedback for the Drafter if REVISE, or explanation for the action.",
    )
    new_search_query: Optional[str] = Field(
        default=None,
        description="The precise new search query if RE_RETRIEVE.",
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="The approved, formatted final answer if ACCEPT or ABSTAIN.",
    )
    is_abstained: bool = Field(
        default=False,
        description="True if the core answer is missing completely.",
    )


class P5Ver1_Pipeline(P3_Pipeline):
    def __init__(self, pipeline_version: str, config: Any, openai_client: Any, logger: Any, run_id: str):
        super().__init__(
            pipeline_version=pipeline_version,
            config=config,
            openai_client=openai_client,
            logger=logger,
            run_id=run_id,
        )
        self.logger.info(
            "P5 critic instruction map loaded | path=%s entries=%s",
            CRITIC_MAP_PATH,
            len(CRITIC_MAP),
        )

    def retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        if self._active_collection is None:
            raise RuntimeError("No active Chroma collection set for retrieval node.")

        loop_count = int(state.get("loop_count", 0))
        question = str(state["question"])
        query = str(state.get("new_search_query", "")).strip() if loop_count > 0 else question
        if not query:
            query = question

        top_k = int(self.config.retrieval_params.top_k) if loop_count == 0 else 3
        n_results = max(top_k * 2, top_k)
        question_id = str(state.get("question_id", ""))
        paper_id = int(state.get("paper_id", 0))

        vectors, embedding_tokens = self.embed_texts(
            self.config.retrieval_params.embedding_model,
            [query],
        )
        query_embedding: Sequence[float] = vectors[0] if vectors else []
        if not query_embedding:
            return {
                "retrieved_chunks": list(state.get("retrieved_chunks", [])),
                "embedding_tokens": int(state.get("embedding_tokens", 0)),
            }

        candidates = self._run_adaptive_hybrid_retrieval(
            query=query,
            query_embedding=query_embedding,
            question_id=question_id,
            paper_id=paper_id,
            n_results=n_results,
        )

        boost_rules = self.boost_map.get(question_id, {})
        ranked_new: List[Dict[str, Any]] = []
        for chunk in candidates:
            metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata", {}), Mapping) else {}
            boost_multiplier = 1.0
            for meta_key, meta_boost_val in boost_rules.items():
                if metadata.get(meta_key) is True:
                    boost_multiplier *= float(meta_boost_val)
            ranked_new.append(
                {
                    "chunk_id": str(chunk.get("chunk_id", "")),
                    "text": str(chunk.get("text", "")),
                    "score": float(chunk.get("score", 0.0)) * boost_multiplier,
                    "metadata": dict(metadata),
                }
            )

        ranked_new.sort(key=lambda item: (float(item.get("score", 0.0)), item.get("chunk_id", "")), reverse=True)
        selected_new = ranked_new[:top_k]

        merged_by_id: Dict[str, Dict[str, Any]] = {}
        for chunk in state.get("retrieved_chunks", []):
            if not isinstance(chunk, dict):
                continue
            chunk_id = str(chunk.get("chunk_id", ""))
            if not chunk_id:
                continue
            merged_by_id[chunk_id] = {
                "chunk_id": chunk_id,
                "text": str(chunk.get("text", "")),
                "score": float(chunk.get("score", 0.0)),
                "metadata": chunk.get("metadata", {}) or {},
            }

        for chunk in selected_new:
            chunk_id = str(chunk.get("chunk_id", ""))
            if not chunk_id:
                continue
            existing = merged_by_id.get(chunk_id)
            if existing is None or float(chunk.get("score", 0.0)) > float(existing.get("score", 0.0)):
                merged_by_id[chunk_id] = chunk

        merged_chunks = sorted(
            merged_by_id.values(),
            key=lambda item: (float(item.get("score", 0.0)), item.get("chunk_id", "")),
            reverse=True,
        )

        self.logger.info(
            "[Q%s] 🔎 Retrieval executed | Loop: %s | Query: '%s' | Total unique chunks now: %s",
            state.get("question_id"),
            loop_count,
            query,
            len(merged_chunks),
        )

        return {
            "retrieved_chunks": merged_chunks,
            "embedding_tokens": int(state.get("embedding_tokens", 0)) + int(embedding_tokens),
        }

    def generate_draft_node(self, state: AgentState) -> Dict[str, Any]:
        question = str(state["question"])
        question_type = str(state.get("question_type", "FREE_TEXT")).upper()
        retrieved_context = "\n\n".join(str(chunk.get("text", "")) for chunk in state.get("retrieved_chunks", []))
        critic_feedback = str(state.get("critic_feedback", "")).strip()

        system_prompt = (
            "You are a precise chemistry research assistant. "
            "Answer ONLY based on the provided retrieved chunks. "
            "Return clean outputs that strictly match the requested schema."
        )
        feedback_block = ""
        if critic_feedback:
            feedback_block = (
                "The Critic rejected your previous draft with this feedback:\n"
                f"{critic_feedback}\n\n"
                "Please revise your answer.\n\n"
            )

        user_prompt = (
            f"Question Type:\n{question_type}\n\n"
            f"Question:\n{question}\n\n"
            f"{feedback_block}"
            f"Retrieved Chunks:\n{retrieved_context}\n"
        )

        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model=self.config.llm_params.model_name,
                temperature=self.config.llm_params.temperature,
                max_tokens=self.config.llm_params.max_tokens,
                response_format=P5DraftResponse,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except AuthenticationError as exc:
            raise RuntimeError("OpenAI authentication error while generating answer. Check OPENAI_API_KEY.") from exc
        except RateLimitError as exc:
            raise RuntimeError("OpenAI quota/rate-limit error while generating answer.") from exc
        except PermissionDeniedError as exc:
            raise RuntimeError(
                f"OpenAI permission error while generating answer: {self._format_api_status_error(exc)}"
            ) from exc
        except APIStatusError as exc:
            raise RuntimeError(
                f"OpenAI status error while generating answer: {self._format_api_status_error(exc)}"
            ) from exc
        except APIError as exc:
            raise RuntimeError(f"OpenAI API error while generating answer: {exc}") from exc

        message = completion.choices[0].message if completion.choices else None
        parsed = getattr(message, "parsed", None)
        if parsed is None:
            raise RuntimeError("P5 draft generation failed: missing parsed response.")

        usage = getattr(completion, "usage", None)
        llm_input_tokens = int(state.get("llm_input_tokens", 0)) + self._usage_tokens(usage, "prompt_tokens")
        llm_output_tokens = int(state.get("llm_output_tokens", 0)) + self._usage_tokens(usage, "completion_tokens")

        return {
            "draft_answer": str(parsed.draft_answer).strip(),
            "reasoning": str(parsed.reasoning).strip(),
            "evidence_quotes": [str(item).strip() for item in parsed.quotes if str(item).strip()],
            "llm_input_tokens": llm_input_tokens,
            "llm_output_tokens": llm_output_tokens,
        }

    def critique_node(self, state: AgentState) -> Dict[str, Any]:
        question_id = str(state.get("question_id", ""))
        specific_instruction = CRITIC_MAP.get(
            question_id,
            "Format the draft answer accurately based on the question.",
        )
        question = str(state["question"])
        question_type = str(state.get("question_type", "FREE_TEXT")).upper()
        draft_answer = str(state.get("draft_answer", ""))
        loop_count = int(state.get("loop_count", 0))
        critic_model = getattr(self.config.llm_params, "critic_model_name", "gpt-4o")

        system_prompt = (
            "You are a rigorous Scientific QA Critic for chemistry RAG. "
            "Evaluate the draft answer and decide the next action.\n\n"
            f"QUESTION-SPECIFIC INSTRUCTION:\n{specific_instruction}\n\n"
            "You must output one status:\n"
            "- ACCEPT: draft is correct and sufficiently complete.\n"
            "- REVISE: draft has formatting or logic issues that can be fixed from existing retrieved chunks.\n"
            "- RE_RETRIEVE: draft lacks required evidence; provide a precise better search query.\n"
            "- ABSTAIN: core answer is truly unavailable after attempts.\n"
            "Do not hallucinate. Keep feedback actionable and concise."
        )
        user_prompt = (
            f"Loop Count:\n{loop_count}\n\n"
            f"Question Type:\n{question_type}\n\n"
            f"Question:\n{question}\n\n"
            f"Draft Answer:\n{draft_answer}\n"
        )

        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model=critic_model,
                temperature=0.0,
                max_tokens=self.config.llm_params.max_tokens,
                response_format=P5CriticResponse,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except AuthenticationError as exc:
            raise RuntimeError("OpenAI authentication error while generating answer. Check OPENAI_API_KEY.") from exc
        except RateLimitError as exc:
            raise RuntimeError("OpenAI quota/rate-limit error while generating answer.") from exc
        except PermissionDeniedError as exc:
            raise RuntimeError(
                f"OpenAI permission error while generating answer: {self._format_api_status_error(exc)}"
            ) from exc
        except APIStatusError as exc:
            raise RuntimeError(
                f"OpenAI status error while generating answer: {self._format_api_status_error(exc)}"
            ) from exc
        except APIError as exc:
            raise RuntimeError(f"OpenAI API error while generating answer: {exc}") from exc

        message = completion.choices[0].message if completion.choices else None
        parsed = getattr(message, "parsed", None)
        if parsed is None:
            raise RuntimeError("P5 critic generation failed: missing parsed response.")

        feedback = str(parsed.feedback).strip() if parsed.feedback is not None else ""
        status = str(parsed.status).strip().upper()
        new_search_query = str(parsed.new_search_query).strip() if parsed.new_search_query is not None else ""
        final_answer = str(parsed.final_answer).strip() if parsed.final_answer is not None else ""
        is_abstained = bool(parsed.is_abstained)

        self.logger.info(
            "[Q%s] 🔄 Critic Status: %s | Loop: %s/2 | Feedback: %s",
            state.get("question_id"),
            status,
            loop_count,
            feedback,
        )

        usage = getattr(completion, "usage", None)
        llm_input_tokens = int(state.get("llm_input_tokens", 0)) + self._usage_tokens(usage, "prompt_tokens")
        llm_output_tokens = int(state.get("llm_output_tokens", 0)) + self._usage_tokens(usage, "completion_tokens")

        next_loop_count = loop_count + 1
        output: Dict[str, Any] = {
            "critic_status": status,
            "critic_feedback": feedback,
            "new_search_query": new_search_query,
            "critique_logic": feedback,
            "is_abstained": is_abstained,
            "loop_count": next_loop_count,
            "llm_input_tokens": llm_input_tokens,
            "llm_output_tokens": llm_output_tokens,
        }
        if status in {"ACCEPT", "ABSTAIN"}:
            output["model_answer"] = final_answer or draft_answer
        return output

    def route_after_critique(self, state: AgentState) -> str:
        status = str(state.get("critic_status", "ACCEPT")).upper()
        loop_count = int(state.get("loop_count", 0))

        if loop_count >= 2 and status in ["REVISE", "RE_RETRIEVE"]:
            self.logger.info(
                "[Q%s] 🛑 Max loops reached (2). Forcing END.",
                state.get("question_id"),
            )
            return END

        if status == "RE_RETRIEVE":
            return "retrieve_node"
        if status == "REVISE":
            return "generate_draft_node"

        return END

    def build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("retrieve_node", self.retrieve_node)
        graph.add_node("generate_draft_node", self.generate_draft_node)
        graph.add_node("critique_node", self.critique_node)

        graph.add_edge(START, "retrieve_node")
        graph.add_edge("retrieve_node", "generate_draft_node")
        graph.add_edge("generate_draft_node", "critique_node")
        graph.add_conditional_edges("critique_node", self.route_after_critique)
        return graph.compile()
