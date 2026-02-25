"""P3 implementation using adaptive multi-query retrieval and structured evidence generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai import APIError, APIStatusError, AuthenticationError, PermissionDeniedError, RateLimitError
from pydantic import BaseModel

from src.pipelines.p2_imp_pipeline import P2_Imp_Pipeline, RRF_K
from src.state import AgentState


class P3Response(BaseModel):
    answer: str
    reasoning: str
    quotes: List[str]


class P3_Pipeline(P2_Imp_Pipeline):
    def __init__(self, pipeline_version: str, config: Any, openai_client: Any, logger: Any, run_id: str):
        super().__init__(
            pipeline_version=pipeline_version,
            config=config,
            openai_client=openai_client,
            logger=logger,
            run_id=run_id,
        )
        self.query_expansion_map_path = Path("specs/query_expansion_map.json")
        self.query_expansion_map = self._load_query_expansion_map(self.query_expansion_map_path)
        self.logger.info(
            "P3 query expansion map loaded | path=%s entries=%s",
            self.query_expansion_map_path,
            len(self.query_expansion_map),
        )

    @staticmethod
    def _load_query_expansion_map(path: Path) -> Dict[str, List[str]]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}

        normalized: Dict[str, List[str]] = {}
        for question_id, expansions in payload.items():
            if not isinstance(expansions, list):
                continue
            cleaned = [str(item).strip() for item in expansions if str(item).strip()]
            normalized[str(question_id)] = cleaned
        return normalized

    def _expanded_queries(self, question: str, question_id: str) -> List[str]:
        expansions = self.query_expansion_map.get(question_id, [])
        raw = [str(question).strip()] + [str(item).strip() for item in expansions[:2]]
        seen = set()
        unique: List[str] = []
        for item in raw:
            if not item or item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique or [str(question)]

    def _run_adaptive_hybrid_retrieval(
        self,
        *,
        query: str,
        query_embedding: Sequence[float],
        question_id: str,
        paper_id: int,
        n_results: int,
    ) -> List[Dict[str, Any]]:
        boost_rules = self.boost_map.get(question_id, {})
        use_bm25 = len(boost_rules) > 0
        alpha = 0.75

        dense_candidates = self._dense_candidates(query_embedding=list(query_embedding), n_results=n_results)

        fused: Dict[str, Dict[str, Any]] = {}
        for rank, candidate in enumerate(dense_candidates, start=1):
            chunk_id = str(candidate.get("chunk_id", ""))
            if not chunk_id:
                continue
            entry = fused.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "text": str(candidate.get("text", "")),
                    "metadata": candidate.get("metadata", {}) or {},
                    "base_rrf_score": 0.0,
                },
            )
            weight = alpha if use_bm25 else 1.0
            entry["base_rrf_score"] += weight * (1.0 / (RRF_K + rank))

        if use_bm25:
            sparse_candidates = self._sparse_candidates(query=query, n_results=n_results, paper_id=paper_id)
            for rank, candidate in enumerate(sparse_candidates, start=1):
                chunk_id = str(candidate.get("chunk_id", ""))
                if not chunk_id:
                    continue
                entry = fused.setdefault(
                    chunk_id,
                    {
                        "chunk_id": chunk_id,
                        "text": str(candidate.get("text", "")),
                        "metadata": candidate.get("metadata", {}) or {},
                        "base_rrf_score": 0.0,
                    },
                )
                if not entry.get("text"):
                    entry["text"] = str(candidate.get("text", ""))
                if not entry.get("metadata"):
                    entry["metadata"] = candidate.get("metadata", {}) or {}
                entry["base_rrf_score"] += (1.0 - alpha) * (1.0 / (RRF_K + rank))

        candidates: List[Dict[str, Any]] = []
        for chunk in fused.values():
            metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata", {}), Mapping) else {}
            candidates.append(
                {
                    "chunk_id": str(chunk.get("chunk_id", "")),
                    "text": str(chunk.get("text", "")),
                    "score": float(chunk.get("base_rrf_score", 0.0)),
                    "metadata": dict(metadata),
                }
            )
        return candidates

    def retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        if self._active_collection is None:
            raise RuntimeError("No active Chroma collection set for retrieval node.")

        top_k = 3
        n_results = max(int(self.config.retrieval_params.top_k) * 2, top_k)
        question = str(state["question"])
        question_id = str(state.get("question_id", ""))
        paper_id = int(state.get("paper_id", 0))

        queries = self._expanded_queries(question=question, question_id=question_id)
        vectors, embedding_tokens = self.embed_texts(
            self.config.retrieval_params.embedding_model,
            queries,
        )

        merged_by_chunk_id: Dict[str, Dict[str, Any]] = {}
        for idx, query in enumerate(queries):
            query_vector = vectors[idx] if idx < len(vectors) else []
            if not query_vector:
                continue
            query_candidates = self._run_adaptive_hybrid_retrieval(
                query=query,
                query_embedding=query_vector,
                question_id=question_id,
                paper_id=paper_id,
                n_results=n_results,
            )
            for candidate in query_candidates:
                chunk_id = str(candidate.get("chunk_id", ""))
                if not chunk_id:
                    continue
                existing = merged_by_chunk_id.get(chunk_id)
                if existing is None or float(candidate.get("score", 0.0)) > float(existing.get("score", 0.0)):
                    merged_by_chunk_id[chunk_id] = {
                        "chunk_id": chunk_id,
                        "text": str(candidate.get("text", "")),
                        "metadata": candidate.get("metadata", {}) or {},
                        "score": float(candidate.get("score", 0.0)),
                    }

        boost_rules = self.boost_map.get(question_id, {})
        ranked_chunks: List[Dict[str, Any]] = []
        for chunk in merged_by_chunk_id.values():
            metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata", {}), Mapping) else {}
            boost_multiplier = 1.0
            for meta_key, meta_boost_val in boost_rules.items():
                if metadata.get(meta_key) is True:
                    boost_multiplier *= float(meta_boost_val)
            final_score = float(chunk.get("score", 0.0)) * boost_multiplier
            ranked_chunks.append(
                {
                    "chunk_id": str(chunk.get("chunk_id", "")),
                    "text": str(chunk.get("text", "")),
                    "score": final_score,
                    "metadata": dict(metadata),
                }
            )

        ranked_chunks.sort(key=lambda item: (float(item.get("score", 0.0)), item.get("chunk_id", "")), reverse=True)
        return {
            "retrieved_chunks": ranked_chunks[:top_k],
            "embedding_tokens": embedding_tokens,
        }

    def generate_node(self, state: AgentState) -> Dict[str, Any]:
        question = str(state["question"])
        question_type = str(state.get("question_type", "FREE_TEXT")).upper()
        retrieved_context = "\n\n".join(str(chunk.get("text", "")) for chunk in state.get("retrieved_chunks", []))

        system_prompt = (
            "You are a chemistry research assistant. "
            "Answer ONLY based on the provided 3 chunks. "
            "Return clean outputs that strictly match the requested schema. "
            "For NUMERIC types: provide only the value and unit (e.g., '180 °C'). "
            "For LIST types: provide a comma-separated list. "
            "Populate 'quotes' with exact sentences from the text."
        )
        user_prompt = (
            f"Question Type:\n{question_type}\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved Chunks (Top 3):\n{retrieved_context}\n"
        )

        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model=self.config.llm_params.model_name,
                temperature=self.config.llm_params.temperature,
                max_tokens=self.config.llm_params.max_tokens,
                response_format=P3Response,
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
            raise RuntimeError("P3 structured generation failed: missing parsed response.")

        answer = str(parsed.answer).strip()
        reasoning = str(parsed.reasoning).strip()
        evidence_quotes = [str(item).strip() for item in parsed.quotes if str(item).strip()]

        usage = getattr(completion, "usage", None)
        llm_input_tokens = self._usage_tokens(usage, "prompt_tokens")
        llm_output_tokens = self._usage_tokens(usage, "completion_tokens")

        return {
            "model_answer": answer,
            "reasoning": reasoning,
            "evidence_quotes": evidence_quotes,
            "llm_input_tokens": llm_input_tokens,
            "llm_output_tokens": llm_output_tokens,
        }

    def build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("retrieve_node", self.retrieve_node)
        graph.add_node("generate_node", self.generate_node)
        graph.add_edge(START, "retrieve_node")
        graph.add_edge("retrieve_node", "generate_node")
        graph.add_edge("generate_node", END)
        return graph.compile()
