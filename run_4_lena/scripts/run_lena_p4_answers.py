"""Run P4 for Lena run 4 without evaluator output or old output paths."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence, Tuple


RUN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = RUN_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chromadb
from openai import (
    APIError,
    APIStatusError,
    AuthenticationError,
    LengthFinishReasonError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from prepare_lena_gold import DEFAULT_GOLD_OUT, DEFAULT_MANIFEST_OUT, DEFAULT_XLSX, convert_lena_workbook
from src.config import OPENAI_API_KEY, RunnerConfig, load_config
from src.pipelines import p4_pipeline as p4_module
from src.utils.document_manager import (
    get_or_build_chroma_collection,
    initialize_document_manager,
    reset_rebuilt_papers,
)


DEFAULT_CONFIG = RUN_DIR / "config_lena_p4_run4.yaml"
DEFAULT_ARTICLES_DIR = RUN_DIR / "articles_4_lena"
DEFAULT_CRITIC_MAP = RUN_DIR / "specs_lena" / "critic_instructions_map_lena_run4.json"
DEFAULT_QUERY_MAP = RUN_DIR / "specs_lena" / "query_expansion_map_lena_run4.json"
DEFAULT_BOOST_MAP = RUN_DIR / "specs_lena" / "metadata_boost_map_lena_run4.json"
LENGTH_ERROR_STATUS = "generation_error_length_limit"
LENGTH_RETRY_FLOOR_TOKENS = 4096


class LenaP4Pipeline(p4_module.P4_Pipeline):
    """P4 pipeline with Lena-local retrieval and critic instruction maps."""

    def __init__(
        self,
        *,
        pipeline_version: str,
        config: Any,
        openai_client: Any,
        logger: Any,
        run_id: str,
        query_map_path: Path,
        boost_map_path: Path,
    ) -> None:
        super().__init__(
            pipeline_version=pipeline_version,
            config=config,
            openai_client=openai_client,
            logger=logger,
            run_id=run_id,
        )
        self.query_expansion_map_path = query_map_path
        self.query_expansion_map = self._load_query_expansion_map(query_map_path)
        self.boost_map_path = boost_map_path
        self.boost_map = self._load_boost_map(boost_map_path)
        self.logger.info(
            "Lena local maps loaded | query_map=%s query_entries=%s boost_map=%s boost_entries=%s",
            query_map_path,
            len(self.query_expansion_map),
            boost_map_path,
            len(self.boost_map),
        )

    def _length_retry_budgets(self) -> List[int]:
        base_tokens = max(1, int(self.config.llm_params.max_tokens))
        retry_tokens = max(base_tokens * 2, LENGTH_RETRY_FLOOR_TOKENS)
        return [base_tokens] if retry_tokens == base_tokens else [base_tokens, retry_tokens]

    def _usage_from_length_error(self, exc: LengthFinishReasonError) -> Tuple[int, int]:
        completion = getattr(exc, "completion", None)
        usage = getattr(completion, "usage", None)
        return (
            self._usage_tokens(usage, "prompt_tokens"),
            self._usage_tokens(usage, "completion_tokens"),
        )

    def _raise_generation_error(self, exc: Exception) -> RuntimeError:
        if isinstance(exc, AuthenticationError):
            return RuntimeError("OpenAI authentication error while generating answer. Check OPENAI_API_KEY.")
        if isinstance(exc, RateLimitError):
            return RuntimeError("OpenAI quota/rate-limit error while generating answer.")
        if isinstance(exc, PermissionDeniedError):
            return RuntimeError(f"OpenAI permission error while generating answer: {self._format_api_status_error(exc)}")
        if isinstance(exc, APIStatusError):
            return RuntimeError(f"OpenAI status error while generating answer: {self._format_api_status_error(exc)}")
        if isinstance(exc, APIError):
            return RuntimeError(f"OpenAI API error while generating answer: {exc}")
        return RuntimeError(f"OpenAI generation error: {exc}")

    def _parse_with_length_retry(
        self,
        *,
        stage: str,
        state: Dict[str, Any],
        response_format: Any,
        build_messages: Any,
    ) -> Tuple[Any | None, int, int, List[Dict[str, Any]]]:
        total_input_tokens = 0
        total_output_tokens = 0
        retry_history: List[Dict[str, Any]] = []
        budgets = self._length_retry_budgets()

        for attempt_index, max_tokens in enumerate(budgets):
            attempt_number = attempt_index + 1
            compact_retry = attempt_index > 0
            try:
                completion = self.openai_client.beta.chat.completions.parse(
                    model=self.config.llm_params.model_name,
                    temperature=self.config.llm_params.temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    messages=build_messages(compact_retry),
                )
            except LengthFinishReasonError as exc:
                input_tokens, output_tokens = self._usage_from_length_error(exc)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                retry_history.append(
                    {
                        "stage": stage,
                        "event": "length_limit",
                        "attempt": attempt_number,
                        "max_tokens": max_tokens,
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                    }
                )
                if attempt_number >= len(budgets):
                    self.logger.error(
                        "P4 %s length failure after retries | paper=%s qid=%s attempts=%s max_tokens=%s",
                        stage,
                        state.get("paper_id"),
                        state.get("question_id"),
                        attempt_number,
                        max_tokens,
                    )
                    return None, total_input_tokens, total_output_tokens, retry_history

                self.logger.warning(
                    "P4 %s length retry | paper=%s qid=%s attempt=%s max_tokens=%s next_max_tokens=%s",
                    stage,
                    state.get("paper_id"),
                    state.get("question_id"),
                    attempt_number,
                    max_tokens,
                    budgets[attempt_number],
                )
                continue
            except (AuthenticationError, RateLimitError, PermissionDeniedError, APIStatusError, APIError) as exc:
                raise self._raise_generation_error(exc) from exc

            usage = getattr(completion, "usage", None)
            total_input_tokens += self._usage_tokens(usage, "prompt_tokens")
            total_output_tokens += self._usage_tokens(usage, "completion_tokens")
            message = completion.choices[0].message if completion.choices else None
            parsed = getattr(message, "parsed", None)
            if parsed is None:
                raise RuntimeError(f"P4 {stage} generation failed: missing parsed response.")
            if retry_history:
                retry_history.append(
                    {
                        "stage": stage,
                        "event": "retry_success",
                        "attempt": attempt_number,
                        "max_tokens": max_tokens,
                    }
                )
            return parsed, total_input_tokens, total_output_tokens, retry_history

        return None, total_input_tokens, total_output_tokens, retry_history

    @staticmethod
    def _compact_draft_suffix() -> str:
        return (
            "\n\nCOMPACT RETRY RULES:\n"
            "- Keep draft_answer under 80 words.\n"
            "- Keep reasoning under 120 words.\n"
            "- Include at most 3 short evidence quotes.\n"
            "- Return only the requested structured fields."
        )

    @staticmethod
    def _compact_critic_suffix() -> str:
        return (
            "\n\nCOMPACT RETRY RULES:\n"
            "- Keep final_answer as short as possible while preserving the required format.\n"
            "- Keep critique_logic under 80 words.\n"
            "- Return only the requested structured fields."
        )

    def generate_draft_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = str(state["question"])
        question_type = str(state.get("question_type", "FREE_TEXT")).upper()
        retrieved_context = "\n\n".join(str(chunk.get("text", "")) for chunk in state.get("retrieved_chunks", []))

        base_system_prompt = (
            "You are a precise chemistry research assistant. "
            "Answer ONLY based on the provided retrieved chunks. "
            "Return clean outputs that strictly match the requested schema. "
            "CRITICAL FORMATTING RULES:\n"
            "- DO NOT just output isolated numbers or disjointed lists.\n"
            "- Maintain brief context in your answer (e.g., instead of just '69 mg, 10 mg', write 'PbBr2: 69 mg, MXene: 10 mg').\n"
            "- For lifetimes or specific properties, include the label (e.g., 'Average lifetime: 4.5 ns').\n"
            "- Populate 'quotes' with exact, verbatim sentences from the text."
        )
        user_prompt = (
            f"Question Type:\n{question_type}\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved Chunks:\n{retrieved_context}\n"
        )

        def build_messages(compact_retry: bool) -> List[Dict[str, str]]:
            system_prompt = base_system_prompt + (self._compact_draft_suffix() if compact_retry else "")
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        parsed, llm_input_tokens, llm_output_tokens, retry_history = self._parse_with_length_retry(
            stage="draft",
            state=state,
            response_format=p4_module.P4DraftResponse,
            build_messages=build_messages,
        )
        if parsed is None:
            feedback = (
                "Draft generation exceeded the output token limit after retrying with "
                f"max_tokens={self._length_retry_budgets()[-1]}. This question was recorded as a handled failure."
            )
            return {
                "draft_answer": "GENERATION_ERROR: draft response exceeded output token limit after retries.",
                "model_answer": "GENERATION_ERROR: draft response exceeded output token limit after retries.",
                "reasoning": feedback,
                "evidence_quotes": [],
                "critique_logic": "Skipped critic because draft generation did not produce parseable structured output.",
                "critic_status": LENGTH_ERROR_STATUS,
                "critic_feedback": feedback,
                "is_abstained": True,
                "loop_history": retry_history,
                "llm_input_tokens": llm_input_tokens,
                "llm_output_tokens": llm_output_tokens,
            }

        result: Dict[str, Any] = {
            "draft_answer": str(parsed.draft_answer).strip(),
            "reasoning": str(parsed.reasoning).strip(),
            "evidence_quotes": [str(item).strip() for item in parsed.quotes if str(item).strip()],
            "llm_input_tokens": llm_input_tokens,
            "llm_output_tokens": llm_output_tokens,
        }
        if retry_history:
            result["loop_history"] = retry_history
        return result

    def route_after_draft(self, state: Dict[str, Any]) -> str:
        if str(state.get("critic_status", "")) == LENGTH_ERROR_STATUS:
            return "bypass_node"
        return super().route_after_draft(state)

    def bypass_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if str(state.get("critic_status", "")) == LENGTH_ERROR_STATUS:
            return {
                "model_answer": str(state.get("model_answer") or state.get("draft_answer", "")),
                "critique_logic": str(state.get("critique_logic", "")),
                "is_abstained": True,
                "critic_status": LENGTH_ERROR_STATUS,
                "critic_feedback": str(state.get("critic_feedback", "")),
                "loop_history": list(state.get("loop_history", [])) if isinstance(state.get("loop_history"), list) else [],
            }
        return super().bypass_node(state)

    def critique_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = str(state["question"])
        question_type = str(state.get("question_type", "FREE_TEXT")).upper()
        draft_answer = str(state.get("draft_answer", ""))
        question_id = str(state.get("question_id", ""))
        specific_instruction = p4_module.CRITIC_MAP.get(
            question_id,
            "Format the draft answer accurately based on the question.",
        )

        base_system_prompt = (
            "You are a rigorous Scientific QA Editor. Your task is to refine a 'Draft Answer' into a pristine "
            "'Final Answer'.\n"
            "You do NOT generate new information. You only filter, format, and correct the Draft.\n\n"
            f"QUESTION-SPECIFIC INSTRUCTION:\n{specific_instruction}\n\n"
            "CRITICAL RULES:\n"
            "1. ABSTENTION: Set `is_abstained = true` ONLY if the core entity or main answer requested by the "
            "question is completely missing, unmeasured, or explicitly stated as not provided. Do NOT abstain if "
            "the main answer is present but partial or secondary information (such as its location in the SI) is "
            "missing.\n"
            "2. ZERO HALLUCINATION: Base your final answer strictly on the provided Draft. Remove all conversational "
            "filler (e.g., 'The document states...').\n"
            "Record your step-by-step logic in 'critique_logic', set the 'is_abstained' flag accurately, and output "
            "the clean 'final_answer'."
        )
        user_prompt = (
            f"Question Type:\n{question_type}\n\n"
            f"Question:\n{question}\n\n"
            f"Draft Answer:\n{draft_answer}\n"
        )

        def build_messages(compact_retry: bool) -> List[Dict[str, str]]:
            system_prompt = base_system_prompt + (self._compact_critic_suffix() if compact_retry else "")
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        parsed, input_tokens, output_tokens, retry_history = self._parse_with_length_retry(
            stage="critic",
            state=state,
            response_format=p4_module.P4FinalResponse,
            build_messages=build_messages,
        )
        prior_history = list(state.get("loop_history", [])) if isinstance(state.get("loop_history"), list) else []
        llm_input_tokens = int(state.get("llm_input_tokens", 0)) + input_tokens
        llm_output_tokens = int(state.get("llm_output_tokens", 0)) + output_tokens
        if parsed is None:
            feedback = (
                "Critic generation exceeded the output token limit after retrying with "
                f"max_tokens={self._length_retry_budgets()[-1]}. The draft answer was kept and this item was marked."
            )
            return {
                "model_answer": draft_answer or "GENERATION_ERROR: critic response exceeded output token limit after retries.",
                "critique_logic": "Critic failed after length-limit retries; using draft answer without critic refinement.",
                "is_abstained": False,
                "critic_status": LENGTH_ERROR_STATUS,
                "critic_feedback": feedback,
                "loop_history": prior_history + retry_history,
                "llm_input_tokens": llm_input_tokens,
                "llm_output_tokens": llm_output_tokens,
            }

        result = {
            "model_answer": str(parsed.final_answer).strip(),
            "critique_logic": str(parsed.critique_logic).strip(),
            "is_abstained": bool(getattr(parsed, "is_abstained", False)),
            "llm_input_tokens": llm_input_tokens,
            "llm_output_tokens": llm_output_tokens,
        }
        if prior_history or retry_history:
            result["loop_history"] = prior_history + retry_history
        return result


def setup_logging(level: str, logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_4_lena_p4")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = logs_dir / f"lena_run4_p4_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    logger.info("log file | path=%s", log_path)
    return logger


def normalize_config_paths(config: RunnerConfig) -> RunnerConfig:
    for field_name in ["data_dir", "chroma_dir", "output_dir"]:
        raw_path = Path(getattr(config.paths, field_name))
        if not raw_path.is_absolute():
            setattr(config.paths, field_name, str((REPO_ROOT / raw_path).resolve()))
    return config


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return [dict(item) for item in payload if isinstance(item, dict)]


def resolve_repo_relative_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def selected_record_keys(selected: Dict[int, List[Dict[str, Any]]]) -> set[Tuple[int, int]]:
    return {
        (int(paper_id), int(item["question_id"]))
        for paper_id, rows in selected.items()
        for item in rows
    }


def load_existing_output_index(path: Path) -> Tuple[set[Tuple[int, int]], str | None]:
    processed: set[Tuple[int, int]] = set()
    run_id: str | None = None
    if not path.exists():
        return processed, run_id

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}") from exc
            if not isinstance(record, dict):
                continue
            if run_id is None and record.get("run_id"):
                run_id = str(record["run_id"])
            try:
                processed.add((int(record["paper_id"]), int(record["question_id"])))
            except (KeyError, TypeError, ValueError):
                continue
    return processed, run_id


def output_needs_leading_newline(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        return handle.read(1) != b"\n"


def ensure_prepared_gold(gold_path: Path, manifest_path: Path, xlsx_path: Path) -> None:
    if gold_path.exists() and manifest_path.exists():
        return
    convert_lena_workbook(xlsx_path=xlsx_path, gold_out=gold_path, manifest_out=manifest_path)


def filter_records(records: Sequence[Dict[str, Any]], config: RunnerConfig) -> Dict[int, List[Dict[str, Any]]]:
    by_paper: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        by_paper.setdefault(int(record["paper_id"]), []).append(dict(record))

    if config.run_control.paper_ids:
        paper_ids = [int(pid) for pid in config.run_control.paper_ids if int(pid) in by_paper]
    else:
        paper_ids = sorted(by_paper)
        if config.run_control.max_papers is not None:
            paper_ids = paper_ids[: int(config.run_control.max_papers)]

    selected: Dict[int, List[Dict[str, Any]]] = {}
    for paper_id in paper_ids:
        questions = sorted(by_paper[paper_id], key=lambda item: int(item["question_id"]))
        if config.run_control.question_ids:
            allowed = {int(qid) for qid in config.run_control.question_ids}
            questions = [item for item in questions if int(item["question_id"]) in allowed]
        elif config.run_control.max_questions_per_paper is not None:
            questions = questions[: int(config.run_control.max_questions_per_paper)]
        if questions:
            selected[paper_id] = questions
    return selected


def _normal_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def resolve_pdf_path(paper: Dict[str, Any], articles_dir: Path) -> Path | None:
    paper_id = int(paper["paper_id"])
    source_name = str(paper.get("source_file_name", "")).strip()
    candidates = [articles_dir / f"paper_{paper_id:02d}.pdf"]
    if source_name:
        source_path = Path(source_name)
        candidates.append(articles_dir / source_path.name)
        if source_path.suffix.lower() != ".pdf":
            candidates.append(articles_dir / f"{source_path.name}.pdf")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    available = [path for path in articles_dir.glob("*.pdf") if path.is_file()]
    if source_name:
        wanted_values = {
            _normal_name(source_name),
            _normal_name(Path(source_name).stem),
            _normal_name(f"{source_name}.pdf"),
        }
        for path in available:
            if _normal_name(path.name) in wanted_values or _normal_name(path.stem) in wanted_values:
                return path
    return None


def validate_article_inputs(
    selected: Dict[int, List[Dict[str, Any]]],
    manifest: Sequence[Dict[str, Any]],
    articles_dir: Path,
) -> Tuple[Dict[int, Path], List[Dict[str, Any]]]:
    by_paper_id = {int(item["paper_id"]): dict(item) for item in manifest}
    resolved: Dict[int, Path] = {}
    missing: List[Dict[str, Any]] = []
    for paper_id in sorted(selected):
        paper = by_paper_id.get(paper_id, {"paper_id": paper_id, "source_file_name": ""})
        pdf_path = resolve_pdf_path(paper, articles_dir)
        if pdf_path is None:
            missing.append(paper)
        else:
            resolved[paper_id] = pdf_path
    return resolved, missing


def build_output_filename(timestamp: str, selected: Dict[int, List[Dict[str, Any]]]) -> str:
    paper_ids = sorted(selected)
    question_ids = sorted({int(item["question_id"]) for rows in selected.values() for item in rows})
    total_questions = sum(len(rows) for rows in selected.values())
    paper_tag = f"p{paper_ids[0]:02d}-{paper_ids[-1]:02d}" if paper_ids else "pnone"
    question_tag = f"q{question_ids[0]}-{question_ids[-1]}" if question_ids else "qnone"
    return f"lena_run4_p4_answers_{timestamp}_papers{len(paper_ids)}_{paper_tag}_{question_tag}_nq{total_questions}.jsonl"


def patch_lena_critic_map(critic_map_path: Path) -> None:
    p4_module.CRITIC_MAP_PATH = critic_map_path
    p4_module.CRITIC_MAP = p4_module._load_critic_map(critic_map_path)


def run_lena_p4(
    *,
    config_path: Path,
    gold_path: Path,
    manifest_path: Path,
    xlsx_path: Path,
    resume_output_path: Path | None = None,
    check_inputs_only: bool = False,
    build_viewer: bool = False,
) -> Path | None:
    ensure_prepared_gold(gold_path=gold_path, manifest_path=manifest_path, xlsx_path=xlsx_path)
    config = normalize_config_paths(load_config(config_path))
    logger = setup_logging(config.logging.level, RUN_DIR / "logs_lena_p4")

    records = load_json_list(gold_path)
    manifest = load_json_list(manifest_path)
    selected = filter_records(records, config)
    total_questions = sum(len(items) for items in selected.values())
    if total_questions == 0:
        raise RuntimeError("No questions selected after applying run_control filters.")

    articles_dir = Path(config.paths.data_dir)
    pdf_paths, missing = validate_article_inputs(selected, manifest, articles_dir)
    logger.info("selected run subset | papers=%s questions=%s articles_dir=%s", len(selected), total_questions, articles_dir)

    if missing:
        logger.error("Missing %s required article PDF files.", len(missing))
        for item in missing[:20]:
            logger.error(
                "missing paper_id=%s label=%s expected=%s",
                item.get("paper_id"),
                item.get("paper_label", ""),
                item.get("expected_pdf_names", []),
            )
        if len(missing) > 20:
            logger.error("missing list truncated | remaining=%s", len(missing) - 20)
        if check_inputs_only:
            return None
        raise FileNotFoundError(
            f"Missing {len(missing)} PDFs under {articles_dir}. "
            "Run with --check-inputs-only for the expected names."
        )

    if check_inputs_only:
        logger.info("input check passed | resolved_pdfs=%s", len(pdf_paths))
        return None

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_run4_lena_P4_answers"
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_mode = "w"
    selected_keys = selected_record_keys(selected)
    processed_keys: set[Tuple[int, int]] = set()
    if resume_output_path is not None:
        output_path = resolve_repo_relative_path(resume_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing_keys, existing_run_id = load_existing_output_index(output_path)
        processed_keys = existing_keys & selected_keys
        ignored_existing = len(existing_keys - selected_keys)
        run_id = existing_run_id or run_id
        output_mode = "a"
        logger.info(
            "resume output | path=%s existing_records=%s selected_existing=%s ignored_existing=%s remaining=%s",
            output_path,
            len(existing_keys),
            len(processed_keys),
            ignored_existing,
            total_questions - len(processed_keys),
        )
    else:
        output_path = output_dir / build_output_filename(timestamp=timestamp, selected=selected)

    patch_lena_critic_map(DEFAULT_CRITIC_MAP)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    chroma_client = chromadb.PersistentClient(path=str(Path(config.paths.chroma_dir)))
    pipeline = LenaP4Pipeline(
        pipeline_version=config.pipeline_version,
        config=config,
        openai_client=openai_client,
        logger=logger,
        run_id=run_id,
        query_map_path=DEFAULT_QUERY_MAP,
        boost_map_path=DEFAULT_BOOST_MAP,
    )

    initialize_document_manager(
        chroma_client=chroma_client,
        embed_texts_fn=pipeline.embed_texts,
        logger=logger,
    )
    reset_rebuilt_papers()

    processed = len(processed_keys)
    run_start = perf_counter()
    needs_leading_newline = output_mode == "a" and output_needs_leading_newline(output_path)
    with output_path.open(output_mode, encoding="utf-8") as out_handle:
        if needs_leading_newline:
            out_handle.write("\n")
        for paper_id in sorted(selected):
            pdf_path = pdf_paths[paper_id]
            collection, paper_build_tokens = get_or_build_chroma_collection(
                paper_id=paper_id,
                pdf_path=pdf_path,
                config=config,
            )
            for gold_item in selected[paper_id]:
                question_id = int(gold_item["question_id"])
                record_key = (paper_id, question_id)
                if record_key in processed_keys:
                    logger.info("resume skip | paper=%s qid=%s", paper_id, question_id)
                    continue

                logger.info("P4 start | paper=%s qid=%s", paper_id, question_id)
                record = pipeline.run(
                    gold_item=gold_item,
                    collection=collection,
                    paper_build_tokens=paper_build_tokens,
                )
                generation = record.get("generation", {})
                if isinstance(generation, dict) and generation.get("critic_status") == LENGTH_ERROR_STATUS:
                    record["generation_error"] = {
                        "type": "LengthFinishReasonError",
                        "handled": True,
                        "message": str(generation.get("critic_feedback", "")),
                    }
                record["lena_run4"] = {
                    "source_file_name": str(gold_item.get("source_file_name", "")),
                    "paper_label": str(gold_item.get("paper_label", f"Paper {paper_id}")),
                    "article_pdf_path": str(pdf_path),
                    "answers_xlsx": str(xlsx_path),
                    "gold_json": str(gold_path),
                    "evaluator_used": False,
                }
                out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_handle.flush()
                processed_keys.add(record_key)
                processed += 1
                logger.info("P4 done | paper=%s qid=%s output=%s", paper_id, question_id, output_path)
                if processed % config.logging.progress_every == 0 or processed == total_questions:
                    logger.info("progress | processed=%s/%s", processed, total_questions)

    logger.info("P4 run complete | output=%s elapsed_s=%.1f", output_path, perf_counter() - run_start)
    if build_viewer:
        from build_lena_answer_viewer import build_viewer as build_lena_viewer

        viewer_path = build_lena_viewer(input_path=output_path)
        logger.info("P4 viewer written | path=%s", viewer_path)
    else:
        logger.info(
            "Use the web UI to inspect this output | command=python3 -m http.server 8000 "
            "url=http://127.0.0.1:8000/run_4_lena/webui/"
        )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P4 answers only, without evaluator.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument(
        "--resume-output",
        type=Path,
        help="Append to an existing JSONL output and skip paper/question pairs already present.",
    )
    parser.add_argument("--check-inputs-only", action="store_true")
    parser.add_argument("--build-static-viewer", action="store_true")
    parser.add_argument("--no-viewer", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_lena_p4(
        config_path=args.config,
        gold_path=args.gold,
        manifest_path=args.manifest,
        xlsx_path=args.xlsx,
        resume_output_path=args.resume_output,
        check_inputs_only=args.check_inputs_only,
        build_viewer=bool(args.build_static_viewer and not args.no_viewer),
    )
    if output_path is not None:
        print(f"Wrote P4 answers to: {output_path}")


if __name__ == "__main__":
    main()
