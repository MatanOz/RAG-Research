"""Main orchestrator for graph-based RAG pipeline runs."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from openai import OpenAI

from src.config import DEFAULT_CONFIG_PATH, OPENAI_API_KEY, RunnerConfig, load_config
from src.pipelines import PIPELINE_REGISTRY
from src.utils.document_manager import (
    get_or_build_chroma_collection,
    initialize_document_manager,
    reset_rebuilt_papers,
)

DEFAULT_GOLD_PATH = Path("specs/gold_master_v4_text_plus_ids.json")
def setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("p0_runner")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_gold_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Gold file must be a JSON list: {path}")
    return payload


def filter_gold_records(records: List[Dict[str, Any]], config: RunnerConfig) -> Dict[int, List[Dict[str, Any]]]:
    by_paper: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        paper_id = int(record["paper_id"])
        by_paper.setdefault(paper_id, []).append(record)

    available_papers = sorted(by_paper.keys())

    if config.run_control.paper_ids:
        selected_papers = [pid for pid in config.run_control.paper_ids if pid in by_paper]
    else:
        selected_papers = available_papers
        if config.run_control.max_papers is not None:
            selected_papers = selected_papers[: config.run_control.max_papers]

    selected: Dict[int, List[Dict[str, Any]]] = {}
    for paper_id in selected_papers:
        paper_questions = sorted(by_paper[paper_id], key=lambda item: int(item["question_id"]))
        if config.run_control.question_ids:
            allowed = set(config.run_control.question_ids)
            paper_questions = [item for item in paper_questions if int(item["question_id"]) in allowed]
        elif config.run_control.max_questions_per_paper is not None:
            paper_questions = paper_questions[: config.run_control.max_questions_per_paper]

        if paper_questions:
            selected[paper_id] = paper_questions

    return selected


def run_pipeline(config_path: Path = DEFAULT_CONFIG_PATH, gold_path: Path = DEFAULT_GOLD_PATH) -> Path:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    config = load_config(config_path)
    logger = setup_logging(config.logging.level)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{config.pipeline_version}"

    logger.info("start run | run_id=%s pipeline_name=%s_chroma_baseline", run_id, config.pipeline_version)
    logger.info(
        "config loaded | embedding_model=%s llm_model=%s chunk_size=%s overlap=%s top_k=%s batch=%s",
        config.retrieval_params.embedding_model,
        config.llm_params.model_name,
        config.retrieval_params.chunk_size,
        config.retrieval_params.chunk_overlap,
        config.retrieval_params.top_k,
        config.retrieval_params.embedding_batch_size,
    )

    gold_records = load_gold_records(gold_path)
    logger.info("gold loaded | total_records=%s", len(gold_records))

    selected = filter_gold_records(gold_records, config)
    total_questions = sum(len(items) for items in selected.values())
    logger.info("subset applied | papers=%s questions=%s", len(selected), total_questions)
    if total_questions == 0:
        raise RuntimeError("No questions selected after applying run_control filters.")

    output_dir = Path("outputs") / config.pipeline_version
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"run_{timestamp}.jsonl"

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    chroma_client = chromadb.PersistentClient(path=str(Path(config.paths.chroma_dir)))

    pipeline_class = PIPELINE_REGISTRY.get(config.pipeline_version)
    if pipeline_class is None:
        raise ValueError(f"Unsupported pipeline_version '{config.pipeline_version}'. Expected one of: {sorted(PIPELINE_REGISTRY.keys())}")

    pipeline = pipeline_class(
        pipeline_version=config.pipeline_version,
        config=config,
        openai_client=openai_client,
        logger=logger,
        run_id=run_id,
    )

    initialize_document_manager(
        chroma_client=chroma_client,
        embed_texts_fn=pipeline.embed_texts,
        logger=logger,
    )
    reset_rebuilt_papers()

    processed = 0
    with output_path.open("w", encoding="utf-8") as out_handle:
        for paper_id in sorted(selected.keys()):
            pdf_path = Path(config.paths.data_dir) / f"paper_{paper_id:02d}.pdf"
            collection, paper_build_tokens = get_or_build_chroma_collection(
                paper_id=paper_id,
                pdf_path=pdf_path,
                config=config,
            )

            for gold_item in selected[paper_id]:
                question_id = int(gold_item["question_id"])
                logger.info("paper=%s qid=%s retrieval start", paper_id, question_id)
                logger.info("paper=%s qid=%s generation start", paper_id, question_id)

                record = pipeline.run(
                    gold_item=gold_item,
                    collection=collection,
                    paper_build_tokens=paper_build_tokens,
                )

                scores = record["retrieval"]["retrieval_scores"]
                score_preview = ",".join(f"{float(score):.4f}" for score in scores[:3])
                logger.info(
                    "paper=%s qid=%s retrieval end | hits=%s top_scores=%s",
                    paper_id,
                    question_id,
                    len(scores),
                    score_preview,
                )
                logger.info("paper=%s qid=%s generation end", paper_id, question_id)

                out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_handle.flush()

                processed += 1
                logger.info("paper=%s qid=%s output write | path=%s", paper_id, question_id, output_path)
                if processed % config.logging.progress_every == 0 or processed == total_questions:
                    logger.info("progress | processed=%s/%s", processed, total_questions)

    logger.info("run complete | output=%s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph-based pipeline (P0/P1/P2)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = run_pipeline(config_path=args.config, gold_path=args.gold)
    print(f"Wrote baseline run to: {path}")
