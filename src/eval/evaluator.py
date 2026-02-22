"""Main orchestration for offline evaluation across multiple pipeline runs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml
from dotenv import load_dotenv

from src.eval.judges.semantic_judge import SemanticJudge
from src.eval.metrics.cost import estimate_question_cost
from src.eval.metrics.retrieval import evaluate_retrieval
from src.eval.metrics.rule_based import compute_rule_based_score


QuestionKey = Tuple[int, int]


DEFAULT_EVAL_CONFIG: Dict[str, Any] = {
    "paths": {
        "gold_path": "specs/gold_master_v4_text_plus_ids.json",
        "output_path": "outputs/eval/ui_dashboard_data.json",
    },
    "runs": {},
    "evaluation": {
        "default_k": 3,
        "numeric_tolerance": 0.05,
        "hallucination_threshold": 0.5,
        "qa_lambda": {
            "NUMERIC": 1.0,
            "LIST": 0.5,
            "STRING": 0.0,
            "CATEGORICAL": 0.0,
            "FREE_TEXT": 0.0,
            "DEFAULT": 0.0,
        },
        "final_weights": {
            "w_qa": 0.4,
            "w_gr": 0.25,
            "w_ret": 0.2,
            "w_eff": 0.15,
        },
    },
    "pricing": {
        "embedding": {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13},
        "llm": {"gpt-4o-mini": {"input_per_million": 0.15, "output_per_million": 0.6}},
    },
    "judge": {
        "enabled": False,
        "model": "gpt-4o",
        "temperature": 0.0,
        "max_tokens": 350,
    },
    "logging": {"level": "INFO"},
}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_eval_config(config_path: Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Evaluation config root must be a mapping: {path}")
    return _deep_merge(DEFAULT_EVAL_CONFIG, raw)


def _setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("evaluation_module")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _min_max_normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


class Evaluator:
    def __init__(self, config_path: Path) -> None:
        load_dotenv()
        self.config_path = Path(config_path)
        self.config = load_eval_config(self.config_path)
        self.logger = _setup_logger(self.config.get("logging", {}).get("level", "INFO"))

        judge_cfg = self.config.get("judge", {})
        self.semantic_judge = SemanticJudge(
            enabled=bool(judge_cfg.get("enabled", False)),
            model=str(judge_cfg.get("model", "gpt-4o")),
            temperature=float(judge_cfg.get("temperature", 0.0)),
            max_tokens=int(judge_cfg.get("max_tokens", 350)),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            logger=self.logger,
        )

        gold_path = Path(self.config.get("paths", {}).get("gold_path", "specs/gold_master_v4_text_plus_ids.json"))
        self.gold_records = self._load_gold(gold_path)
        self.gold_by_key, self.evidence_lookup = self._build_gold_indices(self.gold_records)

    def _load_gold(self, gold_path: Path) -> List[Dict[str, Any]]:
        if not gold_path.exists():
            raise FileNotFoundError(f"Gold dataset not found: {gold_path}")
        with gold_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Gold dataset must be a JSON list: {gold_path}")
        return payload

    def _build_gold_indices(self, records: List[Dict[str, Any]]) -> Tuple[Dict[QuestionKey, Dict[str, Any]], Dict[str, str]]:
        by_key: Dict[QuestionKey, Dict[str, Any]] = {}
        evidence_lookup: Dict[str, str] = {}
        for row in records:
            key = (int(row.get("paper_id", 0)), int(row.get("question_id", 0)))
            by_key[key] = row
            evidence_text = str(row.get("gold_evidence_text", ""))
            for para_id in row.get("evidence_para_ids", []) or []:
                para_id_str = str(para_id)
                if para_id_str and para_id_str not in evidence_lookup:
                    evidence_lookup[para_id_str] = evidence_text
        return by_key, evidence_lookup

    def _resolve_run_path(self, run_path: str) -> Path:
        candidate = Path(run_path)
        if candidate.exists():
            return candidate

        if "*" in run_path:
            matches = sorted(Path(".").glob(run_path), key=lambda p: p.stat().st_mtime)
            if matches:
                return matches[-1]

        if candidate.name == "run_latest.jsonl":
            parent = candidate.parent if str(candidate.parent) != "" else Path(".")
            matches = sorted(parent.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
            if matches:
                return matches[-1]

        raise FileNotFoundError(f"Run JSONL path not found or unresolved: {run_path}")

    def _load_jsonl_records(self, path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.logger.warning("Skipping invalid JSON line %s in %s: %s", line_number, path, exc)
                    continue
                if isinstance(row, dict):
                    records.append(row)
        return records

    def _index_pipeline_records(self, rows: List[Dict[str, Any]]) -> Dict[QuestionKey, Dict[str, Any]]:
        indexed: Dict[QuestionKey, Dict[str, Any]] = {}
        for row in rows:
            key = (int(row.get("paper_id", 0)), int(row.get("question_id", 0)))
            indexed[key] = row
        return indexed

    @staticmethod
    def _first_non_empty(values: List[str]) -> str:
        for value in values:
            if value:
                return value
        return ""

    def _extract_retrieved_context(self, record: Mapping[str, Any]) -> str:
        retrieval = record.get("retrieval", {}) if isinstance(record, dict) else {}
        context = str(retrieval.get("retrieved_context", ""))
        if context:
            return context
        top_chunks = retrieval.get("top_chunks", [])
        if isinstance(top_chunks, list):
            return "\n\n".join(str(chunk.get("text", "")) for chunk in top_chunks if isinstance(chunk, dict))
        return ""

    def evaluate(self) -> Dict[str, Any]:
        run_map = self.config.get("runs", {})
        if not isinstance(run_map, dict) or not run_map:
            raise ValueError("config.runs must define at least one pipeline label and run path")

        indexed_runs: Dict[str, Dict[QuestionKey, Dict[str, Any]]] = {}
        pipeline_labels: List[str] = []
        for label, raw_path in run_map.items():
            try:
                resolved_path = self._resolve_run_path(str(raw_path))
                rows = self._load_jsonl_records(resolved_path)
                if not rows:
                    self.logger.warning("Skipping %s: no records found in %s", label, resolved_path)
                    continue
                indexed_runs[label] = self._index_pipeline_records(rows)
                pipeline_labels.append(label)
                self.logger.info("Loaded pipeline %s from %s (%s records)", label, resolved_path, len(rows))
            except Exception as exc:
                self.logger.warning("Skipping %s due to load error: %s", label, exc)

        if not indexed_runs:
            raise RuntimeError("No valid run files were loaded. Check config.runs paths.")

        all_keys = sorted({key for pipeline_map in indexed_runs.values() for key in pipeline_map.keys()})
        eval_cfg = self.config.get("evaluation", {})
        qa_lambda = eval_cfg.get("qa_lambda", {})
        tolerance = float(eval_cfg.get("numeric_tolerance", 0.05))
        default_k = int(eval_cfg.get("default_k", 3))
        hallucination_threshold = float(eval_cfg.get("hallucination_threshold", 0.5))
        pricing = self.config.get("pricing", {})

        aggregates: Dict[str, Dict[str, Any]] = {
            label: {
                "qa_scores": [],
                "groundedness": [],
                "retrieval_scores": [],
                "latency_ms": [],
                "cost_usd": [],
                "hallucinations": 0,
                "count": 0,
            }
            for label in pipeline_labels
        }
        per_question_comparisons: List[Dict[str, Any]] = []

        for paper_id, question_id in all_keys:
            gold_item = self.gold_by_key.get((paper_id, question_id), {})
            question_type = self._first_non_empty(
                [str(gold_item.get("question_type", ""))]
                + [str(indexed_runs[label].get((paper_id, question_id), {}).get("question_type", "")) for label in pipeline_labels]
            )
            question_text = self._first_non_empty(
                [str(gold_item.get("question", ""))]
                + [str(indexed_runs[label].get((paper_id, question_id), {}).get("question", "")) for label in pipeline_labels]
            )
            gold_answer = self._first_non_empty(
                [str(gold_item.get("reference_answer", ""))]
                + [str(indexed_runs[label].get((paper_id, question_id), {}).get("reference_answer", "")) for label in pipeline_labels]
            )

            row: Dict[str, Any] = {
                "question_id": int(question_id),
                "question_type": str(question_type or "FREE_TEXT"),
                "question_text": question_text,
                "gold_answer": gold_answer,
                "pipelines": {},
            }

            for label in pipeline_labels:
                record = indexed_runs[label].get((paper_id, question_id))
                if record is None:
                    continue

                qtype = str(record.get("question_type", question_type or "FREE_TEXT")).upper()
                model_answer = str(record.get("generation", {}).get("model_answer", ""))
                retrieved_context = self._extract_retrieved_context(record)

                rule_based_score, _ = compute_rule_based_score(
                    question_type=qtype,
                    gold_answer=gold_answer,
                    model_answer=model_answer,
                    numeric_tolerance=tolerance,
                )
                retrieval_eval = evaluate_retrieval(
                    record=record,
                    gold_record=gold_item,
                    evidence_lookup=self.evidence_lookup,
                    k=default_k,
                )
                retrieval_score = float(retrieval_eval.get("recall_at_k", 0.0))

                judge_result = self.semantic_judge.evaluate(
                    question=question_text,
                    question_type=qtype,
                    gold_answer=gold_answer,
                    model_answer=model_answer,
                    retrieved_context=retrieved_context,
                )
                semantic_correctness = float(judge_result.get("semantic_correctness", 0.0))
                groundedness = float(judge_result.get("groundedness", 0.0))
                judge_explanation = str(judge_result.get("explanation", ""))

                lambda_weight = float(qa_lambda.get(qtype, qa_lambda.get("DEFAULT", 0.0)))
                rb_score = float(rule_based_score) if rule_based_score is not None else 0.0
                qa_score = (lambda_weight * rb_score) + ((1.0 - lambda_weight) * semantic_correctness)

                cost_eval = estimate_question_cost(record=record, pricing=pricing)
                latency_ms = float(record.get("logs", {}).get("latency_ms", 0.0))
                cost_usd = float(cost_eval.get("total_eval_cost_usd", record.get("logs", {}).get("estimated_cost_usd", 0.0)))

                row["pipelines"][label] = {
                    "model_answer": model_answer,
                    "qa_score": round(qa_score, 6),
                    "groundedness": round(groundedness, 6),
                    "judge_explanation": judge_explanation,
                    "latency_ms": round(latency_ms, 3),
                    "cost_usd": round(cost_usd, 8),
                }

                agg = aggregates[label]
                agg["qa_scores"].append(qa_score)
                agg["groundedness"].append(groundedness)
                agg["retrieval_scores"].append(retrieval_score)
                agg["latency_ms"].append(latency_ms)
                agg["cost_usd"].append(cost_usd)
                agg["count"] += 1
                if groundedness < hallucination_threshold:
                    agg["hallucinations"] += 1

            if row["pipelines"]:
                per_question_comparisons.append(row)

        interim: Dict[str, Dict[str, float]] = {}
        for label, agg in aggregates.items():
            count = agg["count"] if agg["count"] > 0 else 1
            interim[label] = {
                "qa_micro_score": _mean(agg["qa_scores"]),
                "mean_groundedness": _mean(agg["groundedness"]),
                "mean_retrieval_score": _mean(agg["retrieval_scores"]),
                "mean_latency_ms": _mean(agg["latency_ms"]),
                "mean_cost_usd": _mean(agg["cost_usd"]),
                "hallucination_rate": agg["hallucinations"] / count,
            }

        latencies = [metrics["mean_latency_ms"] for metrics in interim.values()]
        costs = [metrics["mean_cost_usd"] for metrics in interim.values()]
        min_latency, max_latency = min(latencies), max(latencies)
        min_cost, max_cost = min(costs), max(costs)

        final_weights = eval_cfg.get("final_weights", {})
        w_qa = float(final_weights.get("w_qa", 0.4))
        w_gr = float(final_weights.get("w_gr", 0.25))
        w_ret = float(final_weights.get("w_ret", 0.2))
        w_eff = float(final_weights.get("w_eff", 0.15))

        summary_metrics: Dict[str, Dict[str, float]] = {}
        for label, metrics in interim.items():
            norm_latency = _min_max_normalize(metrics["mean_latency_ms"], min_latency, max_latency)
            norm_cost = _min_max_normalize(metrics["mean_cost_usd"], min_cost, max_cost)
            efficiency_score = 0.5 * (1.0 - norm_latency) + 0.5 * (1.0 - norm_cost)
            final_score = (
                (w_qa * metrics["qa_micro_score"])
                + (w_gr * metrics["mean_groundedness"])
                + (w_ret * metrics["mean_retrieval_score"])
                + (w_eff * efficiency_score)
            )
            summary_metrics[label] = {
                "final_score": round(final_score, 6),
                "qa_micro_score": round(metrics["qa_micro_score"], 6),
                "mean_groundedness": round(metrics["mean_groundedness"], 6),
                "mean_cost_usd": round(metrics["mean_cost_usd"], 8),
                "hallucination_rate": round(metrics["hallucination_rate"], 6),
            }

        return {
            "summary_metrics": summary_metrics,
            "per_question_comparisons": per_question_comparisons,
        }

