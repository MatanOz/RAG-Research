"""P0 baseline runner with persistent ChromaDB retrieval and config-driven controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import chromadb
import fitz
import yaml
from dotenv import load_dotenv
from openai import APIError, APIStatusError, AuthenticationError, OpenAI, RateLimitError
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_PATH = Path("configs/p0_baseline.yaml")
DEFAULT_GOLD_PATH = Path("specs/gold_master_v4_text_plus_ids.json")

CHROMA_CLIENT: Any = None
OPENAI_CLIENT: Optional[OpenAI] = None
LOGGER: Optional[logging.Logger] = None
REBUILT_PAPERS_THIS_RUN: set = set()


class RetrievalParams(BaseModel):
    embedding_model: str
    chunking_method: str = "fixed_char"
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    top_k: int = Field(ge=1)
    embedding_batch_size: int = Field(ge=1)

    model_config = ConfigDict(extra="forbid")


class LLMParams(BaseModel):
    model_name: str
    temperature: float
    max_tokens: int = Field(gt=0)

    model_config = ConfigDict(extra="forbid")


class PathsConfig(BaseModel):
    data_dir: str
    chroma_dir: str
    output_dir: str

    model_config = ConfigDict(extra="forbid")


class RunControl(BaseModel):
    paper_ids: Optional[List[int]] = None
    max_papers: Optional[int] = Field(default=None, ge=1)
    question_ids: Optional[List[int]] = None
    max_questions_per_paper: Optional[int] = Field(default=None, ge=1)

    model_config = ConfigDict(extra="forbid")


class LoggingConfig(BaseModel):
    level: str = "INFO"
    progress_every: int = Field(default=10, ge=1)

    model_config = ConfigDict(extra="forbid")


class RunnerConfig(BaseModel):
    project_name: str
    pipeline_version: str
    retrieval_params: RetrievalParams
    llm_params: LLMParams
    paths: PathsConfig
    run_control: RunControl = Field(default_factory=RunControl)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = ConfigDict(extra="forbid")


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


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


def setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("p0_runner")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_runner_config(config_path: Path) -> RunnerConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return RunnerConfig.model_validate(raw)


def load_gold_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Gold file must be a JSON list: {path}")
    return payload


def stable_collection_name(paper_id: int) -> str:
    return f"paper_{paper_id:02d}"


def get_pdf_signature(pdf_path: Path) -> Dict[str, str]:
    try:
        digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        return {"type": "sha256", "value": digest}
    except OSError:
        stat = pdf_path.stat()
        fallback = f"{int(stat.st_mtime)}:{stat.st_size}"
        return {"type": "mtime_size", "value": fallback}


def fingerprint_path(chroma_dir: Path, paper_id: int) -> Path:
    return chroma_dir / "meta" / f"paper_{paper_id:02d}_fingerprint.json"


def build_fingerprint(paper_id: int, pdf_path: Path, config: RunnerConfig) -> Dict[str, Any]:
    return {
        "paper_id": paper_id,
        "embedding_model": config.retrieval_params.embedding_model,
        "chunking": {
            "method": config.retrieval_params.chunking_method,
            "chunk_size": config.retrieval_params.chunk_size,
            "overlap": config.retrieval_params.chunk_overlap,
        },
        "pdf_path": str(pdf_path),
        "pdf_signature": get_pdf_signature(pdf_path),
    }


def read_fingerprint(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_fingerprint(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def extract_pdf_pages(pdf_path: Path) -> List[str]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pages: List[str] = []
    with fitz.open(pdf_path) as document:
        for page in document:
            pages.append((page.get_text("text") or "").strip())
    return pages


def chunk_pdf_pages(paper_id: int, page_texts: List[str], chunk_size: int, chunk_overlap: int) -> List[ChunkRecord]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    full_parts: List[str] = []
    page_ranges: List[Tuple[int, int, int]] = []
    cursor = 0
    for page_num, text in enumerate(page_texts, start=1):
        if full_parts:
            full_parts.append("\n")
            cursor += 1
        start = cursor
        full_parts.append(text)
        cursor += len(text)
        end = cursor
        page_ranges.append((page_num, start, end))

    full_text = "".join(full_parts)
    if not full_text:
        return [
            ChunkRecord(
                chunk_id=f"paper_{paper_id:02d}_chunk_00000",
                text="",
                metadata={"paper_id": paper_id, "chunk_index": 0},
            )
        ]

    step = chunk_size - chunk_overlap
    chunks: List[ChunkRecord] = []
    chunk_index = 0
    for start in range(0, len(full_text), step):
        end = min(len(full_text), start + chunk_size)
        text = full_text[start:end]
        overlap_pages = [p for (p, p_start, p_end) in page_ranges if not (p_end <= start or p_start >= end)]

        metadata: Dict[str, Any] = {"paper_id": paper_id, "chunk_index": chunk_index}
        if overlap_pages:
            metadata["page_start"] = min(overlap_pages)
            metadata["page_end"] = max(overlap_pages)

        chunks.append(
            ChunkRecord(
                chunk_id=f"paper_{paper_id:02d}_chunk_{chunk_index:05d}",
                text=text,
                metadata=metadata,
            )
        )
        chunk_index += 1
        if end >= len(full_text):
            break
    return chunks


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for offset in range(0, len(items), batch_size):
        yield items[offset : offset + batch_size]


def usage_tokens(usage: Any, key: str) -> int:
    if usage is None:
        return 0
    value = usage.get(key, 0) if isinstance(usage, dict) else getattr(usage, key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def embed_texts(client: OpenAI, model: str, texts: Sequence[str]) -> Tuple[List[List[float]], int]:
    try:
        response = client.embeddings.create(model=model, input=list(texts))
    except AuthenticationError as exc:
        raise RuntimeError("OpenAI authentication error while creating embeddings. Check OPENAI_API_KEY.") from exc
    except RateLimitError as exc:
        raise RuntimeError("OpenAI quota/rate-limit error while creating embeddings.") from exc
    except APIStatusError as exc:
        raise RuntimeError(f"OpenAI status error while creating embeddings: {exc.status_code}") from exc
    except APIError as exc:
        raise RuntimeError(f"OpenAI API error while creating embeddings: {exc}") from exc

    vectors = [row.embedding for row in response.data]
    tokens = usage_tokens(getattr(response, "usage", None), "prompt_tokens")
    if tokens == 0:
        tokens = usage_tokens(getattr(response, "usage", None), "total_tokens")
    return vectors, tokens


def generate_answer(
    client: OpenAI,
    model_name: str,
    temperature: float,
    max_tokens: int,
    question: str,
    retrieved_context: str,
) -> Tuple[str, int, int]:
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
        completion = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
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
    prompt_tokens = usage_tokens(usage, "prompt_tokens")
    completion_tokens = usage_tokens(usage, "completion_tokens")
    return text.strip(), prompt_tokens, completion_tokens


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


def require_runtime_context() -> Tuple[Any, OpenAI, logging.Logger, set]:
    if CHROMA_CLIENT is None or OPENAI_CLIENT is None or LOGGER is None:
        raise RuntimeError("Runtime context is not initialized.")
    return CHROMA_CLIENT, OPENAI_CLIENT, LOGGER, REBUILT_PAPERS_THIS_RUN


def get_or_build_chroma_collection(paper_id: int, pdf_path: Path, config: RunnerConfig):
    chroma_client, openai_client, logger, rebuilt_papers = require_runtime_context()
    chroma_dir = Path(config.paths.chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    (chroma_dir / "meta").mkdir(parents=True, exist_ok=True)

    collection_name = stable_collection_name(paper_id)
    fp_path = fingerprint_path(chroma_dir, paper_id)
    current_fp = build_fingerprint(paper_id=paper_id, pdf_path=pdf_path, config=config)
    existing_fp = read_fingerprint(fp_path)

    existing_collection = None
    try:
        existing_collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        existing_collection = None

    needs_rebuild = existing_collection is None or existing_fp != current_fp

    if not needs_rebuild:
        count = existing_collection.count()
        logger.info(
            "paper=%s chroma load | collection=%s chunks=%s persist_dir=%s",
            paper_id,
            collection_name,
            count,
            chroma_dir,
        )
        return existing_collection

    if paper_id in rebuilt_papers:
        logger.info("paper=%s rebuild already performed in this run; reusing collection", paper_id)
        if existing_collection is None:
            existing_collection = chroma_client.get_collection(name=collection_name)
        return existing_collection

    logger.info(
        "paper=%s chroma build | collection=%s reason=%s persist_dir=%s",
        paper_id,
        collection_name,
        "missing" if existing_collection is None else "fingerprint_changed",
        chroma_dir,
    )

    if existing_collection is not None:
        chroma_client.delete_collection(name=collection_name)

    collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    page_texts = extract_pdf_pages(pdf_path)
    chunks = chunk_pdf_pages(
        paper_id=paper_id,
        page_texts=page_texts,
        chunk_size=config.retrieval_params.chunk_size,
        chunk_overlap=config.retrieval_params.chunk_overlap,
    )

    logger.info(
        "paper=%s chunking done | method=%s chunks=%s chunk_size=%s overlap=%s",
        paper_id,
        config.retrieval_params.chunking_method,
        len(chunks),
        config.retrieval_params.chunk_size,
        config.retrieval_params.chunk_overlap,
    )

    batch_size = config.retrieval_params.embedding_batch_size
    batches = list(batched(chunks, batch_size))
    for batch_index, batch in enumerate(batches, start=1):
        texts = [item.text for item in batch]
        ids = [item.chunk_id for item in batch]
        metas = [item.metadata for item in batch]
        vectors, _ = embed_texts(openai_client, config.retrieval_params.embedding_model, texts)
        collection.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=vectors)
        logger.info(
            "paper=%s embedding progress | batch=%s/%s batch_size=%s",
            paper_id,
            batch_index,
            len(batches),
            len(batch),
        )

    write_fingerprint(fp_path, current_fp)
    rebuilt_papers.add(paper_id)
    logger.info("paper=%s chroma build complete | collection=%s chunks=%s", paper_id, collection_name, len(chunks))
    return collection


def filter_gold_records(records: List[Dict[str, Any]], control: RunControl) -> Dict[int, List[Dict[str, Any]]]:
    by_paper: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        paper_id = int(record["paper_id"])
        by_paper.setdefault(paper_id, []).append(record)

    available_papers = sorted(by_paper.keys())

    if control.paper_ids:
        selected_papers = [pid for pid in control.paper_ids if pid in by_paper]
    else:
        selected_papers = available_papers
        if control.max_papers is not None:
            selected_papers = selected_papers[: control.max_papers]

    selected: Dict[int, List[Dict[str, Any]]] = {}
    for paper_id in selected_papers:
        paper_questions = sorted(by_paper[paper_id], key=lambda item: int(item["question_id"]))
        if control.question_ids:
            allowed = set(control.question_ids)
            paper_questions = [item for item in paper_questions if int(item["question_id"]) in allowed]
        elif control.max_questions_per_paper is not None:
            paper_questions = paper_questions[: control.max_questions_per_paper]

        if paper_questions:
            selected[paper_id] = paper_questions

    return selected


def query_chroma(
    collection: Any,
    openai_client: OpenAI,
    question: str,
    embedding_model: str,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], int]:
    vectors, embedding_tokens = embed_texts(openai_client, embedding_model, [question])
    query_embedding = vectors[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
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
        score = 1.0 - distance
        chunks.append(
            {
                "chunk_id": str(chunk_id),
                "text": docs[idx] if idx < len(docs) else "",
                "score": score,
                "metadata": metadata or {},
            }
        )
    return chunks, embedding_tokens


def chunk_para_ids(chunk: Dict[str, Any]) -> List[str]:
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


def build_output_record(
    run_id: str,
    pipeline_name: str,
    config: RunnerConfig,
    gold_item: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    model_answer: str,
    latency_ms: float,
    embedding_tokens: int,
    llm_input_tokens: int,
    llm_output_tokens: int,
) -> Dict[str, Any]:
    top_chunks_for_schema = []
    for rank, chunk in enumerate(retrieved_chunks[:3], start=1):
        top_chunks_for_schema.append(
            {
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "para_ids": chunk_para_ids(chunk),
                "score": float(chunk["score"]),
                "text": chunk["text"],
            }
        )

    tokens_input = embedding_tokens + llm_input_tokens
    tokens_output = llm_output_tokens

    payload = {
        "run_id": run_id,
        "pipeline_name": pipeline_name,
        "paper_id": int(gold_item["paper_id"]),
        "question_id": int(gold_item["question_id"]),
        "question": str(gold_item["question"]),
        "reference_answer": str(gold_item.get("reference_answer", "")),
        "question_type": str(gold_item.get("question_type", "FREE_TEXT")),
        "is_answerable": bool(gold_item.get("is_answerable", False)),
        "gold_evidence_para_ids": [str(x) for x in gold_item.get("evidence_para_ids", [])],
        "models": {
            "embedding_model": config.retrieval_params.embedding_model,
            "llm_model": config.llm_params.model_name,
            "reranker_model": None,
            "judge_model": None,
        },
        "retrieval": {
            "k": config.retrieval_params.top_k,
            "retrieved_para_ids": [chunk["chunk_id"] for chunk in retrieved_chunks],
            "retrieval_scores": [float(chunk["score"]) for chunk in retrieved_chunks],
            "top_chunks": top_chunks_for_schema,
            "retrieved_context": "\n\n".join(chunk["text"] for chunk in retrieved_chunks),
            "hit_at_1": None,
            "hit_at_3": None,
            "hit_at_5": None,
            "mrr": None,
        },
        "generation": {
            "model_answer": model_answer,
        },
        "logs": {
            "latency_ms": round(latency_ms, 3),
            "tokens_input": int(tokens_input),
            "tokens_output": int(tokens_output),
            "total_tokens": int(tokens_input + tokens_output),
            "estimated_cost_usd": estimate_cost_usd(
                embedding_model=config.retrieval_params.embedding_model,
                llm_model=config.llm_params.model_name,
                embedding_input_tokens=embedding_tokens,
                llm_input_tokens=llm_input_tokens,
                llm_output_tokens=llm_output_tokens,
            ),
        },
    }

    validated = PipelineOutputRecord.model_validate(payload)
    return validated.model_dump(mode="json")


def run_p0_baseline(config_path: Path, gold_path: Path) -> Path:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    config = load_runner_config(config_path)

    global LOGGER
    LOGGER = setup_logging(config.logging.level)

    global CHROMA_CLIENT
    chroma_dir = Path(config.paths.chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    CHROMA_CLIENT = chromadb.PersistentClient(path=str(chroma_dir))

    global OPENAI_CLIENT
    OPENAI_CLIENT = OpenAI(api_key=api_key)

    global REBUILT_PAPERS_THIS_RUN
    REBUILT_PAPERS_THIS_RUN = set()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{config.pipeline_version}"
    pipeline_name = f"{config.pipeline_version}_chroma_baseline"

    LOGGER.info("start run | run_id=%s pipeline_name=%s", run_id, pipeline_name)
    LOGGER.info(
        "config loaded | embedding_model=%s llm_model=%s chunk_size=%s overlap=%s top_k=%s batch=%s",
        config.retrieval_params.embedding_model,
        config.llm_params.model_name,
        config.retrieval_params.chunk_size,
        config.retrieval_params.chunk_overlap,
        config.retrieval_params.top_k,
        config.retrieval_params.embedding_batch_size,
    )

    gold_records = load_gold_records(gold_path)
    LOGGER.info("gold loaded | total_records=%s", len(gold_records))

    selected = filter_gold_records(gold_records, config.run_control)
    total_questions = sum(len(items) for items in selected.values())
    LOGGER.info("subset applied | papers=%s questions=%s", len(selected), total_questions)
    if total_questions == 0:
        raise RuntimeError("No questions selected after applying run_control filters.")

    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"run_{timestamp}.jsonl"

    processed = 0
    with output_path.open("w", encoding="utf-8") as out_handle:
        for paper_id in sorted(selected.keys()):
            pdf_path = Path(config.paths.data_dir) / f"paper_{paper_id:02d}.pdf"
            collection = get_or_build_chroma_collection(paper_id, pdf_path, config)

            for gold_item in selected[paper_id]:
                question_id = int(gold_item["question_id"])
                question = str(gold_item["question"])

                LOGGER.info("paper=%s qid=%s retrieval start", paper_id, question_id)
                q_start = perf_counter()

                retrieved_chunks, embedding_tokens = query_chroma(
                    collection=collection,
                    openai_client=OPENAI_CLIENT,
                    question=question,
                    embedding_model=config.retrieval_params.embedding_model,
                    top_k=config.retrieval_params.top_k,
                )

                if not retrieved_chunks:
                    raise RuntimeError(f"No retrieval results for paper={paper_id}, question_id={question_id}")

                score_preview = ",".join(f"{chunk['score']:.4f}" for chunk in retrieved_chunks[:3])
                LOGGER.info(
                    "paper=%s qid=%s retrieval end | hits=%s top_scores=%s",
                    paper_id,
                    question_id,
                    len(retrieved_chunks),
                    score_preview,
                )

                LOGGER.info("paper=%s qid=%s generation start", paper_id, question_id)
                retrieved_context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
                model_answer, llm_input_tokens, llm_output_tokens = generate_answer(
                    client=OPENAI_CLIENT,
                    model_name=config.llm_params.model_name,
                    temperature=config.llm_params.temperature,
                    max_tokens=config.llm_params.max_tokens,
                    question=question,
                    retrieved_context=retrieved_context,
                )
                LOGGER.info("paper=%s qid=%s generation end", paper_id, question_id)

                record = build_output_record(
                    run_id=run_id,
                    pipeline_name=pipeline_name,
                    config=config,
                    gold_item=gold_item,
                    retrieved_chunks=retrieved_chunks,
                    model_answer=model_answer,
                    latency_ms=(perf_counter() - q_start) * 1000.0,
                    embedding_tokens=embedding_tokens,
                    llm_input_tokens=llm_input_tokens,
                    llm_output_tokens=llm_output_tokens,
                )
                out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1

                LOGGER.info("paper=%s qid=%s output write | path=%s", paper_id, question_id, output_path)
                if processed % config.logging.progress_every == 0 or processed == total_questions:
                    LOGGER.info("progress | processed=%s/%s", processed, total_questions)

    LOGGER.info("run complete | output=%s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P0 baseline with ChromaDB")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = run_p0_baseline(config_path=args.config, gold_path=args.gold)
    print(f"Wrote baseline run to: {path}")
