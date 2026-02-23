"""P2 implementation using hybrid retrieval (dense + sparse), RRF, and metadata-aware ranking."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from rank_bm25 import BM25Okapi

from src.base_pipeline import BaseGraphPipeline
from src.state import AgentState

RRF_K = 60.0


class P2_Pipeline(BaseGraphPipeline):
    def __init__(self, pipeline_version: str, config: Any, openai_client: Any, logger: Any, run_id: str):
        self.boost_map_path = Path("specs/metadata_boost_map.json")
        self.boost_map = self._load_boost_map(self.boost_map_path)
        super().__init__(pipeline_version=pipeline_version, config=config, openai_client=openai_client, logger=logger, run_id=run_id)
        self.logger.info(
            "P2 metadata boost map loaded | path=%s entries=%s",
            self.boost_map_path,
            len(self.boost_map),
        )

    @staticmethod
    def _load_boost_map(path: Path) -> Dict[str, Dict[str, float]]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}

        normalized: Dict[str, Dict[str, float]] = {}
        for question_id, rules in payload.items():
            if not isinstance(rules, dict):
                continue
            cast_rules: Dict[str, float] = {}
            for key, value in rules.items():
                try:
                    cast_rules[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            normalized[str(question_id)] = cast_rules
        return normalized

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    def _dense_candidates(self, query_embedding: List[float], n_results: int) -> List[Dict[str, Any]]:
        result = self._active_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        dense: List[Dict[str, Any]] = []
        for idx, chunk_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            dense.append(
                {
                    "chunk_id": str(chunk_id),
                    "text": docs[idx] if idx < len(docs) else "",
                    "metadata": metadata or {},
                    "score": self.distance_to_similarity(distance, "cosine"),
                    "rank": idx + 1,
                }
            )
        return dense

    def _sparse_candidates(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        corpus = self._active_collection.get(include=["documents", "metadatas"])

        ids = corpus.get("ids", []) or []
        docs = corpus.get("documents", []) or []
        metadatas = corpus.get("metadatas", []) or []
        if not ids:
            return []

        tokenized_docs = [self._tokenize(str(doc)) for doc in docs]
        if not any(tokenized_docs):
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(tokenized_query)
        sorted_indexes = sorted(
            range(len(ids)),
            key=lambda idx: float(bm25_scores[idx]),
            reverse=True,
        )

        sparse: List[Dict[str, Any]] = []
        for rank, idx in enumerate(sorted_indexes[:n_results], start=1):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            sparse.append(
                {
                    "chunk_id": str(ids[idx]),
                    "text": docs[idx] if idx < len(docs) else "",
                    "metadata": metadata or {},
                    "score": float(bm25_scores[idx]),
                    "rank": rank,
                }
            )
        return sparse

    def retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        if self._active_collection is None:
            raise RuntimeError("No active Chroma collection set for retrieval node.")

        top_k = int(self.config.retrieval_params.top_k)
        n_results = max(top_k * 2, top_k)

        vectors, embedding_tokens = self.embed_texts(
            self.config.retrieval_params.embedding_model,
            [state["question"]],
        )
        query_embedding = vectors[0]

        dense_candidates = self._dense_candidates(query_embedding=query_embedding, n_results=n_results)
        sparse_candidates = self._sparse_candidates(query=state["question"], n_results=n_results)

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
            entry["base_rrf_score"] += 1.0 / (RRF_K + rank)

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
            entry["base_rrf_score"] += 1.0 / (RRF_K + rank)

        question_id = str(state.get("question_id", ""))
        boost_rules = self.boost_map.get(question_id, {})

        ranked_chunks: List[Dict[str, Any]] = []
        for chunk in fused.values():
            metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata", {}), Mapping) else {}
            boost_multiplier = 1.0
            for meta_key, meta_boost_val in boost_rules.items():
                if metadata.get(meta_key) is True:
                    boost_multiplier *= float(meta_boost_val)
            final_score = float(chunk.get("base_rrf_score", 0.0)) * boost_multiplier
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
        retrieved_context = "\n\n".join(str(chunk.get("text", "")) for chunk in state.get("retrieved_chunks", []))
        model_answer, llm_input_tokens, llm_output_tokens = self.generate_answer(
            question=state["question"],
            retrieved_context=retrieved_context,
        )
        return {
            "model_answer": model_answer,
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
