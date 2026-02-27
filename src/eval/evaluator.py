"""Main orchestration for offline evaluation across multiple pipeline runs."""

from __future__ import annotations

import json
import logging
import os
import re
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
        "default_k": 5,
        "numeric_tolerance": 0.05,
        "hallucination_threshold": 0.5,
        "no_gold_policy": {
            "enabled": True,
            "empty_answer_markers": [
                "",
                "none",
                "null",
                "n/a",
                "na",
                "unknown",
                "not available",
                "not reported",
                "unmeasured",
                "not measured",
            ],
            "status_markers": ["unmeasured", "not reported", "missing", "unknown", "n/a", "na"],
            "abstention_markers": [
                "cannot determine",
                "can't determine",
                "cannot be determined",
                "insufficient context",
                "insufficient information",
                "not provided",
                "not reported",
                "unknown",
                "not available",
                "n/a",
                "not mentioned",
                "does not mention",
                "does not include",
                "does not specify",
                "not specified",
                "not in the context",
                "not enough information",
            ],
            "abstention_qa_score": 1.0,
            "non_abstention_qa_score": 0.0,
            "non_abstention_groundedness_cap": 0.2,
            "force_hallucination_on_non_abstention": True,
            "exclude_retrieval_from_aggregate": True,
        },
        "gold_present_policy": {
            "enabled": True,
            "abstention_qa_cap": 0.0,
            "abstention_semantic_cap": 0.0,
            "abstention_groundedness_cap": 0.2,
            "force_hallucination_on_abstention": True,
            "enforce_numeric_coverage_for_free_text": True,
            "min_gold_numeric_facts": 2,
        },
        "qa_lambda": {
            "NUMERIC": 0.2,
            "LIST": 0.2,
            "STRING": 0.0,
            "CATEGORICAL": 0.0,
            "FREE_TEXT": 0.0,
            "DEFAULT": 0.0,
        },
        "final_weights": {
            "w_qa": 0.6,
            "w_gr": 0.2,
            "w_ret": 0.1,
            "w_eff": 0.1,
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
    "logging": {
        "level": "INFO",
        "progress_every_questions": 20,
        "log_question_details": False,
    },
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
        self.progress_every_questions = max(
            1,
            int(self.config.get("logging", {}).get("progress_every_questions", 20)),
        )
        self.log_question_details = bool(self.config.get("logging", {}).get("log_question_details", False))

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
        self.logger.info(
            "eval init | config=%s gold_records=%s evidence_entries=%s progress_every=%s detailed_logs=%s",
            self.config_path,
            len(self.gold_records),
            len(self.evidence_lookup),
            self.progress_every_questions,
            self.log_question_details,
        )

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
            para_ids = row.get("gold_evidence_para_ids", []) or row.get("evidence_para_ids", []) or []
            for para_id in para_ids:
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

    @staticmethod
    def _contains_marker(text: str, markers: List[str]) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        normalized = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
        if not normalized:
            return False
        for marker in markers:
            marker_text = str(marker or "").strip().lower()
            if not marker_text:
                continue
            marker_norm = re.sub(r"[^a-z0-9]+", " ", marker_text).strip()
            if not marker_norm:
                continue
            if normalized == marker_norm:
                return True
            pattern = r"\b" + re.escape(marker_norm).replace(r"\ ", r"\s+") + r"\b"
            if re.search(pattern, normalized):
                return True
        return False

    def _is_no_gold_case(self, gold_item: Mapping[str, Any], gold_answer: str, policy: Mapping[str, Any]) -> bool:
        if not bool(policy.get("enabled", True)):
            return False

        answer = (gold_answer or "").strip().lower()
        empty_markers = [str(x).strip().lower() for x in policy.get("empty_answer_markers", [])]
        if not answer:
            return True
        if answer in set(empty_markers):
            return True
        if self._contains_marker(answer, empty_markers):
            return True

        is_answerable = gold_item.get("is_answerable")
        if isinstance(is_answerable, bool) and not is_answerable:
            return True

        status_text = str(gold_item.get("status", ""))
        status_markers = [str(x).strip().lower() for x in policy.get("status_markers", [])]
        if self._contains_marker(status_text, status_markers):
            return True
        return False

    def _is_abstention_answer(self, model_answer: str, policy: Mapping[str, Any]) -> bool:
        answer = (model_answer or "").strip().lower()
        if not answer:
            return True
        abstention_markers = [str(x).strip().lower() for x in policy.get("abstention_markers", [])]
        if self._contains_marker(answer, abstention_markers):
            return True
        short_plain = re.sub(r"[\W_]+", "", answer)
        return short_plain in {"na", "none", "unknown"}

    @staticmethod
    def _extract_numeric_literals(text: str) -> List[str]:
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text or "")
        seen = set()
        out: List[str] = []
        for raw in numbers:
            try:
                normalized = str(float(raw))
            except ValueError:
                continue
            if normalized not in seen:
                seen.add(normalized)
                out.append(normalized)
        return out

    def _numeric_fact_coverage(self, gold_answer: str, model_answer: str) -> Dict[str, float]:
        gold_nums = self._extract_numeric_literals(gold_answer)
        pred_nums = set(self._extract_numeric_literals(model_answer))
        if not gold_nums:
            return {"gold_count": 0.0, "matched_count": 0.0, "coverage": 1.0}
        matched = sum(1 for value in gold_nums if value in pred_nums)
        return {
            "gold_count": float(len(gold_nums)),
            "matched_count": float(matched),
            "coverage": float(matched) / float(len(gold_nums)),
        }

    def _extract_retrieved_context(self, record: Mapping[str, Any]) -> str:
        retrieval = record.get("retrieval", {}) if isinstance(record, dict) else {}
        context = str(retrieval.get("retrieved_context", ""))
        if context:
            return context
        top_chunks = retrieval.get("top_chunks", [])
        if isinstance(top_chunks, list):
            return "\n\n".join(str(chunk.get("text", "")) for chunk in top_chunks if isinstance(chunk, dict))
        return ""

    def _extract_retrieved_chunks(self, record: Mapping[str, Any], k: int) -> List[Dict[str, Any]]:
        retrieval = record.get("retrieval", {}) if isinstance(record, dict) else {}
        top_chunks = retrieval.get("top_chunks", [])
        chunks: List[Dict[str, Any]] = []
        if isinstance(top_chunks, list):
            for idx, chunk in enumerate(top_chunks[:k], start=1):
                if not isinstance(chunk, dict):
                    continue
                chunk_row: Dict[str, Any] = {
                    "rank": int(chunk.get("rank", idx)),
                    "chunk_id": str(chunk.get("chunk_id", "")),
                    "score": float(chunk.get("score", 0.0) or 0.0),
                    "text": str(chunk.get("text", "")),
                }
                para_ids = chunk.get("para_ids", [])
                if isinstance(para_ids, list):
                    chunk_row["para_ids"] = [str(x) for x in para_ids]
                chunks.append(chunk_row)
        if chunks:
            return chunks

        fallback_context = str(retrieval.get("retrieved_context", ""))
        if fallback_context:
            return [
                {
                    "rank": 1,
                    "chunk_id": "",
                    "score": 0.0,
                    "text": fallback_context,
                    "para_ids": [],
                }
            ]
        return []

    def evaluate(self) -> Dict[str, Any]:
        run_map = self.config.get("runs", {})
        if not isinstance(run_map, dict) or not run_map:
            raise ValueError("config.runs must define at least one pipeline label and run path")
        self.logger.info("eval stage=start | configured_runs=%s", len(run_map))

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
        self.logger.info("eval stage=runs_loaded | active_pipelines=%s", len(indexed_runs))

        all_keys = sorted({key for pipeline_map in indexed_runs.values() for key in pipeline_map.keys()})
        self.logger.info("eval stage=question_index_ready | unique_questions=%s", len(all_keys))
        eval_cfg = self.config.get("evaluation", {})
        qa_lambda = eval_cfg.get("qa_lambda", {})
        tolerance = float(eval_cfg.get("numeric_tolerance", 0.05))
        default_k = int(eval_cfg.get("default_k", 3))
        hallucination_threshold = float(eval_cfg.get("hallucination_threshold", 0.5))
        no_gold_policy = eval_cfg.get("no_gold_policy", {})
        gold_present_policy = eval_cfg.get("gold_present_policy", {})
        pricing = self.config.get("pricing", {})

        aggregates: Dict[str, Dict[str, Any]] = {
            label: {
                "qa_scores": [],
                "groundedness": [],
                "retrieval_scores": [],
                "latency_ms": [],
                "cost_usd": [],
                "hallucinations": 0,
                "no_gold_cases": 0,
                "no_gold_abstentions": 0,
                "count": 0,
            }
            for label in pipeline_labels
        }
        per_question_comparisons: List[Dict[str, Any]] = []

        total_questions = len(all_keys)
        for question_index, (paper_id, question_id) in enumerate(all_keys, start=1):
            if (
                question_index == 1
                or question_index % self.progress_every_questions == 0
                or question_index == total_questions
            ):
                self.logger.info(
                    "eval progress | question=%s/%s paper=%s question_id=%s",
                    question_index,
                    total_questions,
                    paper_id,
                    question_id,
                )
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
                "paper_id": int(paper_id),
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
                generation_payload = record.get("generation", {})
                if not isinstance(generation_payload, dict):
                    generation_payload = {}

                model_answer = str(generation_payload.get("model_answer", ""))
                raw_reasoning = generation_payload.get("reasoning")
                reasoning = str(raw_reasoning).strip() if raw_reasoning is not None else None
                if reasoning == "":
                    reasoning = None

                raw_critique_logic = generation_payload.get("critique_logic")
                critique_logic = str(raw_critique_logic).strip() if raw_critique_logic is not None else None
                if critique_logic == "":
                    critique_logic = None

                raw_evidence_quotes = generation_payload.get("evidence_quotes", [])
                if isinstance(raw_evidence_quotes, list):
                    evidence_quotes = [str(item).strip() for item in raw_evidence_quotes if str(item).strip()]
                elif raw_evidence_quotes is None:
                    evidence_quotes = []
                else:
                    single_quote = str(raw_evidence_quotes).strip()
                    evidence_quotes = [single_quote] if single_quote else []
                retrieved_context = self._extract_retrieved_context(record)
                retrieved_chunks = self._extract_retrieved_chunks(record, default_k)

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
                gold_evidence_text = str(retrieval_eval.get("gold_evidence_text_used", ""))

                judge_result = self.semantic_judge.evaluate(
                    question=question_text,
                    question_type=qtype,
                    gold_answer=gold_answer,
                    model_answer=model_answer,
                    retrieved_context=retrieved_context,
                )
                semantic_correctness_raw = float(judge_result.get("semantic_correctness", 0.0))
                semantic_correctness = semantic_correctness_raw
                groundedness = float(judge_result.get("groundedness", 0.0))
                judge_explanation = str(judge_result.get("explanation", ""))

                lambda_weight = float(qa_lambda.get(qtype, qa_lambda.get("DEFAULT", 0.0)))
                rb_score = float(rule_based_score) if rule_based_score is not None else 0.0
                qa_score_pre_guard = (lambda_weight * rb_score) + ((1.0 - lambda_weight) * semantic_correctness)
                qa_score = qa_score_pre_guard

                is_no_gold_case = self._is_no_gold_case(gold_item, gold_answer, no_gold_policy)
                is_abstained_field = generation_payload.get("is_abstained")
                if isinstance(is_abstained_field, bool):
                    is_abstained = is_abstained_field
                else:
                    is_abstained = self._is_abstention_answer(model_answer, no_gold_policy)
                forced_hallucination = False
                abstention_penalized_with_gold = False
                numeric_coverage = {"gold_count": 0.0, "matched_count": 0.0, "coverage": 1.0}

                if is_no_gold_case and bool(no_gold_policy.get("enabled", True)):
                    if is_abstained:
                        qa_score = float(no_gold_policy.get("abstention_qa_score", 1.0))
                        judge_explanation = (
                            f"{judge_explanation} | no_gold_policy: abstention accepted"
                            if judge_explanation
                            else "no_gold_policy: abstention accepted"
                        )
                    else:
                        qa_score = float(no_gold_policy.get("non_abstention_qa_score", 0.0))
                        groundedness = min(
                            groundedness,
                            float(no_gold_policy.get("non_abstention_groundedness_cap", 0.2)),
                        )
                        forced_hallucination = bool(
                            no_gold_policy.get("force_hallucination_on_non_abstention", True)
                        )
                        judge_explanation = (
                            f"{judge_explanation} | no_gold_policy: non-abstention penalized"
                            if judge_explanation
                            else "no_gold_policy: non-abstention penalized"
                        )
                elif bool(gold_present_policy.get("enabled", True)) and is_abstained:
                    abstention_penalized_with_gold = True
                    semantic_correctness = min(
                        semantic_correctness,
                        float(gold_present_policy.get("abstention_semantic_cap", 0.0)),
                    )
                    groundedness = min(
                        groundedness,
                        float(gold_present_policy.get("abstention_groundedness_cap", 0.2)),
                    )
                    qa_score = min(qa_score, float(gold_present_policy.get("abstention_qa_cap", 0.0)))
                    forced_hallucination = bool(gold_present_policy.get("force_hallucination_on_abstention", True))
                    judge_explanation = (
                        f"{judge_explanation} | gold_present_policy: abstention penalized"
                        if judge_explanation
                        else "gold_present_policy: abstention penalized"
                    )

                if (
                    not is_no_gold_case
                    and bool(gold_present_policy.get("enabled", True))
                    and bool(gold_present_policy.get("enforce_numeric_coverage_for_free_text", True))
                    and qtype == "FREE_TEXT"
                ):
                    numeric_coverage = self._numeric_fact_coverage(gold_answer, model_answer)
                    min_facts = int(gold_present_policy.get("min_gold_numeric_facts", 2))
                    if numeric_coverage["gold_count"] >= float(min_facts):
                        qa_score = min(qa_score, numeric_coverage["coverage"])
                        semantic_correctness = min(semantic_correctness, numeric_coverage["coverage"])
                        if numeric_coverage["coverage"] < 1.0:
                            judge_explanation = (
                                f"{judge_explanation} | gold_present_policy: numeric coverage {numeric_coverage['matched_count']:.0f}/{numeric_coverage['gold_count']:.0f}"
                                if judge_explanation
                                else f"gold_present_policy: numeric coverage {numeric_coverage['matched_count']:.0f}/{numeric_coverage['gold_count']:.0f}"
                            )

                cost_eval = estimate_question_cost(record=record, pricing=pricing)
                latency_ms = float(record.get("logs", {}).get("latency_ms", 0.0))
                cost_usd = float(cost_eval.get("total_eval_cost_usd", record.get("logs", {}).get("estimated_cost_usd", 0.0)))
                if self.log_question_details:
                    self.logger.info(
                        "eval question | paper=%s question_id=%s pipeline=%s qtype=%s qa=%.3f grounded=%.3f retrieval=%.3f latency_ms=%.1f cost_usd=%.8f",
                        paper_id,
                        question_id,
                        label,
                        qtype,
                        qa_score,
                        groundedness,
                        retrieval_score,
                        latency_ms,
                        cost_usd,
                    )

                row["pipelines"][label] = {
                    "model_answer": model_answer,
                    "reasoning": reasoning,
                    "evidence_quotes": evidence_quotes,
                    "critique_logic": critique_logic,
                    "generation": {
                        "reasoning": reasoning,
                        "evidence_quotes": evidence_quotes,
                        "critique_logic": critique_logic,
                    },
                    "qa_score": round(qa_score, 6),
                    "groundedness": round(groundedness, 6),
                    "retrieval_score": round(retrieval_score, 6),
                    "gold_evidence_text": gold_evidence_text,
                    "retrieved_chunks": retrieved_chunks,
                    "retrieved_context": retrieved_context,
                    "judge_explanation": judge_explanation,
                    "latency_ms": round(latency_ms, 3),
                    "cost_usd": round(cost_usd, 8),
                    "is_no_gold_case": bool(is_no_gold_case),
                    "is_abstained": bool(is_abstained),
                    "debug": {
                        "semantic_correctness_raw": round(semantic_correctness_raw, 6),
                        "semantic_correctness_used": round(semantic_correctness, 6),
                        "rule_based_score": round(rb_score, 6),
                        "lambda_weight": round(lambda_weight, 6),
                        "qa_score_pre_guard": round(qa_score_pre_guard, 6),
                        "qa_score_post_guard": round(qa_score, 6),
                        "abstention_penalized_with_gold": bool(abstention_penalized_with_gold),
                        "numeric_coverage": {
                            "gold_count": int(numeric_coverage["gold_count"]),
                            "matched_count": int(numeric_coverage["matched_count"]),
                            "coverage": round(float(numeric_coverage["coverage"]), 6),
                        },
                    },
                }

                agg = aggregates[label]
                agg["qa_scores"].append(qa_score)
                agg["groundedness"].append(groundedness)
                if not (
                    is_no_gold_case
                    and bool(no_gold_policy.get("exclude_retrieval_from_aggregate", True))
                ):
                    agg["retrieval_scores"].append(retrieval_score)
                agg["latency_ms"].append(latency_ms)
                agg["cost_usd"].append(cost_usd)
                agg["count"] += 1
                if is_no_gold_case:
                    agg["no_gold_cases"] += 1
                    if is_abstained:
                        agg["no_gold_abstentions"] += 1
                if forced_hallucination or groundedness < hallucination_threshold:
                    agg["hallucinations"] += 1

            if row["pipelines"]:
                per_question_comparisons.append(row)

        self.logger.info("eval stage=aggregation_start | pipelines=%s", len(pipeline_labels))
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
                "no_gold_cases": float(agg["no_gold_cases"]),
                "abstention_rate_on_no_gold": (
                    agg["no_gold_abstentions"] / agg["no_gold_cases"] if agg["no_gold_cases"] > 0 else 0.0
                ),
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
                "mean_retrieval_score": round(metrics["mean_retrieval_score"], 6),
                "mean_latency_ms": round(metrics["mean_latency_ms"], 3),
                "mean_cost_usd": round(metrics["mean_cost_usd"], 8),
                "hallucination_rate": round(metrics["hallucination_rate"], 6),
                "no_gold_cases": int(metrics["no_gold_cases"]),
                "abstention_rate_on_no_gold": round(metrics["abstention_rate_on_no_gold"], 6),
            }
            self.logger.info(
                "eval summary | pipeline=%s final=%.4f qa=%.4f grounded=%.4f retrieval=%.4f efficiency=%.4f cost=%.8f latency_ms=%.2f",
                label,
                final_score,
                metrics["qa_micro_score"],
                metrics["mean_groundedness"],
                metrics["mean_retrieval_score"],
                efficiency_score,
                metrics["mean_cost_usd"],
                metrics["mean_latency_ms"],
            )

        self.logger.info(
            "eval stage=complete | pipelines=%s compared_questions=%s",
            len(summary_metrics),
            len(per_question_comparisons),
        )

        return {
            "summary_metrics": summary_metrics,
            "per_question_comparisons": per_question_comparisons,
        }
