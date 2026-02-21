"""Base graph pipeline abstraction shared across RAG pipeline variants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from langgraph.graph.state import CompiledStateGraph
from openai import APIError, APIStatusError, AuthenticationError, OpenAI, RateLimitError
from pydantic import BaseModel, ConfigDict, Field

from src.state import AgentState
from src.utils.cost_estimator import estimate_cost_usd, estimate_embedding_only_cost_usd


class ModelsSection(BaseModel):
    embedding_model: str
    llm_model: str
    reranker_model: Optional[str] = None
    judge_model: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class RetrievalChunk(BaseModel):
    rank: int = Field(ge=1)
    chunk_id: Optional[str] = None
    para_ids: List[str]
    score: float
    text: str

    model_config = ConfigDict(extra="forbid")


class RetrievalSection(BaseModel):
    k: int = Field(ge=1)
    retrieved_para_ids: List[str]
    retrieval_scores: List[float]
    top_chunks: List[RetrievalChunk] = Field(min_length=1, max_length=3)
    retrieved_context: str
    hit_at_1: Optional[bool] = None
    hit_at_3: Optional[bool] = None
    hit_at_5: Optional[bool] = None
    mrr: Optional[float] = None

    model_config = ConfigDict(extra="forbid")


class GenerationSection(BaseModel):
    model_answer: str

    model_config = ConfigDict(extra="forbid")


class LogsSection(BaseModel):
    latency_ms: float = Field(ge=0)
    tokens_input: int = Field(ge=0)
    tokens_output: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    paper_build_embedding_tokens: int = Field(default=0, ge=0)
    paper_build_estimated_cost_usd: float = Field(default=0.0, ge=0)
    paper_build_included_in_estimated_cost: bool = False

    model_config = ConfigDict(extra="forbid")


class PipelineOutputRecord(BaseModel):
    run_id: str
    pipeline_name: str
    paper_id: int
    question_id: int
    question: str
    reference_answer: str
    question_type: Literal["LIST", "NUMERIC", "STRING", "CATEGORICAL", "FREE_TEXT"]
    is_answerable: bool
    gold_evidence_para_ids: List[str]
    models: ModelsSection
    retrieval: RetrievalSection
    generation: GenerationSection
    logs: LogsSection

    model_config = ConfigDict(extra="forbid")


class BaseGraphPipeline(ABC):
    def __init__(self, pipeline_version: str, config: Any, openai_client: OpenAI, logger: Any, run_id: str):
        self.pipeline_version = pipeline_version
        self.config = config
        self.openai_client = openai_client
        self.logger = logger
        self.run_id = run_id
        self.pipeline_name = f"{self.pipeline_version}_chroma_baseline"
        self.output_dir = Path("outputs") / self.pipeline_version
        self._active_collection: Any = None
        self.graph = self.build_graph()

    @abstractmethod
    def build_graph(self) -> CompiledStateGraph:
        raise NotImplementedError

    def run(self, gold_item: Dict[str, Any], collection: Any, paper_build_tokens: int) -> Dict[str, Any]:
        start = perf_counter()
        self._active_collection = collection

        initial_state: AgentState = {
            "question": str(gold_item["question"]),
            "paper_id": int(gold_item["paper_id"]),
            "question_id": int(gold_item["question_id"]),
            "retrieved_chunks": [],
            "model_answer": "",
            "embedding_tokens": 0,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
        }

        final_state = self.graph.invoke(initial_state)
        latency_ms = (perf_counter() - start) * 1000.0
        return self._build_output_record(
            gold_item=gold_item,
            final_state=final_state,
            latency_ms=latency_ms,
            paper_build_embedding_tokens=paper_build_tokens,
        )

    def embed_texts(self, model: str, texts: Sequence[str]) -> Tuple[List[List[float]], int]:
        try:
            response = self.openai_client.embeddings.create(model=model, input=list(texts))
        except AuthenticationError as exc:
            raise RuntimeError("OpenAI authentication error while creating embeddings. Check OPENAI_API_KEY.") from exc
        except RateLimitError as exc:
            raise RuntimeError("OpenAI quota/rate-limit error while creating embeddings.") from exc
        except APIStatusError as exc:
            raise RuntimeError(f"OpenAI status error while creating embeddings: {exc.status_code}") from exc
        except APIError as exc:
            raise RuntimeError(f"OpenAI API error while creating embeddings: {exc}") from exc

        vectors = [row.embedding for row in response.data]
        tokens = self._usage_tokens(getattr(response, "usage", None), "prompt_tokens")
        if tokens == 0:
            tokens = self._usage_tokens(getattr(response, "usage", None), "total_tokens")
        return vectors, tokens

    def generate_answer(self, question: str, retrieved_context: str) -> Tuple[str, int, int]:
        system_prompt = (
            "You are a chemistry research assistant. Answer only from the provided context. "
            "If context is insufficient, say so clearly."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved Context:\n{retrieved_context}\n\n"
            "Return a concise answer."
        )

        try:
            completion = self.openai_client.chat.completions.create(
                model=self.config.llm_params.model_name,
                temperature=self.config.llm_params.temperature,
                max_tokens=self.config.llm_params.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except AuthenticationError as exc:
            raise RuntimeError("OpenAI authentication error while generating answer. Check OPENAI_API_KEY.") from exc
        except RateLimitError as exc:
            raise RuntimeError("OpenAI quota/rate-limit error while generating answer.") from exc
        except APIStatusError as exc:
            raise RuntimeError(f"OpenAI status error while generating answer: {exc.status_code}") from exc
        except APIError as exc:
            raise RuntimeError(f"OpenAI API error while generating answer: {exc}") from exc

        text = completion.choices[0].message.content or ""
        usage = getattr(completion, "usage", None)
        prompt_tokens = self._usage_tokens(usage, "prompt_tokens")
        completion_tokens = self._usage_tokens(usage, "completion_tokens")
        return text.strip(), prompt_tokens, completion_tokens

    @staticmethod
    def distance_to_similarity(distance: float, space: str = "cosine") -> float:
        if space == "cosine":
            if distance <= 1.0:
                return max(0.0, 1.0 - distance)
            return max(0.0, 1.0 - distance / 2.0)
        return max(0.0, 1.0 - distance)

    @staticmethod
    def _usage_tokens(usage: Any, key: str) -> int:
        if usage is None:
            return 0
        value = usage.get(key, 0) if isinstance(usage, dict) else getattr(usage, key, 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _chunk_para_ids(chunk: Dict[str, Any]) -> List[str]:
        metadata = chunk.get("metadata", {})
        para_ids = [chunk["chunk_id"]]
        if "paper_id" in metadata:
            para_ids.append(f"paper_id={metadata['paper_id']}")
        if "chunk_index" in metadata:
            para_ids.append(f"chunk_index={metadata['chunk_index']}")
        if "page_start" in metadata:
            para_ids.append(f"page_start={metadata['page_start']}")
        if "page_end" in metadata:
            para_ids.append(f"page_end={metadata['page_end']}")
        return para_ids

    def _build_output_record(
        self,
        gold_item: Dict[str, Any],
        final_state: Dict[str, Any],
        latency_ms: float,
        paper_build_embedding_tokens: int,
    ) -> Dict[str, Any]:
        retrieved_chunks = list(final_state.get("retrieved_chunks", []))
        top_chunks_for_schema = []
        for rank, chunk in enumerate(retrieved_chunks[:3], start=1):
            top_chunks_for_schema.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.get("chunk_id"),
                    "para_ids": self._chunk_para_ids(chunk),
                    "score": float(chunk.get("score", 0.0)),
                    "text": str(chunk.get("text", "")),
                }
            )

        embedding_tokens = int(final_state.get("embedding_tokens", 0))
        llm_input_tokens = int(final_state.get("llm_input_tokens", 0))
        llm_output_tokens = int(final_state.get("llm_output_tokens", 0))
        tokens_input = embedding_tokens + llm_input_tokens
        tokens_output = llm_output_tokens

        payload = {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "paper_id": int(gold_item["paper_id"]),
            "question_id": int(gold_item["question_id"]),
            "question": str(gold_item["question"]),
            "reference_answer": str(gold_item.get("reference_answer", "")),
            "question_type": str(gold_item.get("question_type", "FREE_TEXT")),
            "is_answerable": bool(gold_item.get("is_answerable", False)),
            "gold_evidence_para_ids": [str(x) for x in gold_item.get("evidence_para_ids", [])],
            "models": {
                "embedding_model": self.config.retrieval_params.embedding_model,
                "llm_model": self.config.llm_params.model_name,
                "reranker_model": None,
                "judge_model": None,
            },
            "retrieval": {
                "k": self.config.retrieval_params.top_k,
                "retrieved_para_ids": [str(chunk.get("chunk_id", "")) for chunk in retrieved_chunks],
                "retrieval_scores": [float(chunk.get("score", 0.0)) for chunk in retrieved_chunks],
                "top_chunks": top_chunks_for_schema,
                "retrieved_context": "\n\n".join(str(chunk.get("text", "")) for chunk in retrieved_chunks),
                "hit_at_1": None,
                "hit_at_3": None,
                "hit_at_5": None,
                "mrr": None,
            },
            "generation": {
                "model_answer": str(final_state.get("model_answer", "")),
            },
            "logs": {
                "latency_ms": round(latency_ms, 3),
                "tokens_input": int(tokens_input),
                "tokens_output": int(tokens_output),
                "total_tokens": int(tokens_input + tokens_output),
                "estimated_cost_usd": estimate_cost_usd(
                    embedding_model=self.config.retrieval_params.embedding_model,
                    llm_model=self.config.llm_params.model_name,
                    embedding_input_tokens=embedding_tokens,
                    llm_input_tokens=llm_input_tokens,
                    llm_output_tokens=llm_output_tokens,
                ),
                "paper_build_embedding_tokens": int(paper_build_embedding_tokens),
                "paper_build_estimated_cost_usd": estimate_embedding_only_cost_usd(
                    self.config.retrieval_params.embedding_model,
                    int(paper_build_embedding_tokens),
                ),
                "paper_build_included_in_estimated_cost": False,
            },
        }

        validated = PipelineOutputRecord.model_validate(payload)
        return validated.model_dump(mode="json")
