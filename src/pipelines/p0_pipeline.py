"""P0 implementation using a linear LangGraph pipeline."""

from __future__ import annotations

from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.base_pipeline import BaseGraphPipeline
from src.state import AgentState


class P0_Pipeline(BaseGraphPipeline):
    def retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        if self._active_collection is None:
            raise RuntimeError("No active Chroma collection set for retrieval node.")

        vectors, embedding_tokens = self.embed_texts(
            self.config.retrieval_params.embedding_model,
            [state["question"]],
        )
        query_embedding = vectors[0]
        result = self._active_collection.query(
            query_embeddings=[query_embedding],
            n_results=self.config.retrieval_params.top_k,
            include=["documents", "distances", "metadatas"],
        )

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        chunks: List[Dict[str, Any]] = []
        for idx, chunk_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            score = self.distance_to_similarity(distance, "cosine")
            chunks.append(
                {
                    "chunk_id": str(chunk_id),
                    "text": docs[idx] if idx < len(docs) else "",
                    "score": score,
                    "metadata": metadata or {},
                }
            )

        return {
            "retrieved_chunks": chunks,
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
