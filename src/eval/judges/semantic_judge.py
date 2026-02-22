"""Semantic judge integration with structured JSON output and robust fallbacks."""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Dict

from openai import APIError, APIStatusError, AuthenticationError, OpenAI, RateLimitError

from src.eval.judges.prompts import get_system_prompt


TOKEN_RE = re.compile(r"[a-z0-9]+")
RATE_LIMIT_RETRY_COUNT = 3
RATE_LIMIT_BACKOFF_SECONDS = [1.0, 2.0, 4.0]
RATE_LIMIT_JITTER_MAX_SECONDS = 0.25
RATE_LIMIT_MAX_WAIT_SECONDS = 2.0


def _clamp_01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _token_f1(reference: str, candidate: str) -> float:
    ref_tokens = TOKEN_RE.findall((reference or "").lower())
    cand_tokens = TOKEN_RE.findall((candidate or "").lower())
    if not ref_tokens and not cand_tokens:
        return 1.0
    if not ref_tokens or not cand_tokens:
        return 0.0

    ref_set = set(ref_tokens)
    cand_set = set(cand_tokens)
    overlap = len(ref_set & cand_set)
    precision = overlap / len(cand_set) if cand_set else 0.0
    recall = overlap / len(ref_set) if ref_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _token_recall(reference: str, candidate: str) -> float:
    ref_tokens = set(TOKEN_RE.findall((reference or "").lower()))
    cand_tokens = set(TOKEN_RE.findall((candidate or "").lower()))
    if not ref_tokens:
        return 0.0
    if not cand_tokens:
        return 0.0
    return len(ref_tokens & cand_tokens) / len(ref_tokens)


class SemanticJudge:
    def __init__(
        self,
        enabled: bool,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
        logger: Any,
    ) -> None:
        self.enabled = enabled
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger
        self.client = OpenAI(api_key=api_key) if enabled and api_key else None
        if self.logger is not None:
            self.logger.info(
                "judge init | enabled=%s model=%s retries=%s backoff=%s jitter_max=%.2fs max_wait=%.2fs",
                self.enabled,
                self.model,
                RATE_LIMIT_RETRY_COUNT,
                RATE_LIMIT_BACKOFF_SECONDS,
                RATE_LIMIT_JITTER_MAX_SECONDS,
                RATE_LIMIT_MAX_WAIT_SECONDS,
            )

    def _retry_sleep_seconds(self, attempt_index: int) -> float:
        base = RATE_LIMIT_BACKOFF_SECONDS[min(attempt_index, len(RATE_LIMIT_BACKOFF_SECONDS) - 1)]
        jitter = random.uniform(0.0, RATE_LIMIT_JITTER_MAX_SECONDS)
        return min(RATE_LIMIT_MAX_WAIT_SECONDS, base + jitter)

    def _fallback(self, question_type: str, gold_answer: str, model_answer: str, retrieved_context: str, reason: str) -> Dict[str, Any]:
        semantic = _token_f1(gold_answer, model_answer)
        groundedness = _token_recall(model_answer, retrieved_context)
        return {
            "semantic_correctness": _clamp_01(semantic),
            "groundedness": _clamp_01(groundedness),
            "explanation": f"fallback_judge: {reason}; lexical scoring used for {question_type}",
        }

    def evaluate(
        self,
        question: str,
        question_type: str,
        gold_answer: str,
        model_answer: str,
        retrieved_context: str,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return self._fallback(question_type, gold_answer, model_answer, retrieved_context, "judge_disabled_in_config")
        if self.client is None:
            return self._fallback(question_type, gold_answer, model_answer, retrieved_context, "missing_openai_api_key")

        system_prompt = get_system_prompt(question_type)
        user_payload = {
            "question": question,
            "question_type": question_type,
            "gold_answer": gold_answer,
            "model_answer": model_answer,
            "retrieved_context": retrieved_context,
        }

        response = None
        for attempt in range(RATE_LIMIT_RETRY_COUNT + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                    ],
                )
                if attempt > 0:
                    self.logger.info(
                        "Judge recovered after rate-limit retries | qtype=%s attempts=%s",
                        question_type,
                        attempt + 1,
                    )
                break
            except AuthenticationError as exc:
                return self._fallback(question_type, gold_answer, model_answer, retrieved_context, f"auth_error:{exc}")
            except RateLimitError as exc:
                if attempt >= RATE_LIMIT_RETRY_COUNT:
                    return self._fallback(question_type, gold_answer, model_answer, retrieved_context, f"rate_limit:{exc}")
                wait_seconds = self._retry_sleep_seconds(attempt)
                self.logger.warning(
                    "Judge rate-limited for %s. retry=%s/%s wait=%.2fs",
                    question_type,
                    attempt + 1,
                    RATE_LIMIT_RETRY_COUNT,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue
            except APIStatusError as exc:
                if int(getattr(exc, "status_code", 0) or 0) == 429:
                    if attempt >= RATE_LIMIT_RETRY_COUNT:
                        return self._fallback(
                            question_type,
                            gold_answer,
                            model_answer,
                            retrieved_context,
                            f"api_status:429:{exc}",
                        )
                    wait_seconds = self._retry_sleep_seconds(attempt)
                    self.logger.warning(
                        "Judge got API 429 for %s. retry=%s/%s wait=%.2fs",
                        question_type,
                        attempt + 1,
                        RATE_LIMIT_RETRY_COUNT,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue
                return self._fallback(question_type, gold_answer, model_answer, retrieved_context, f"api_status:{exc.status_code}")
            except APIError as exc:
                return self._fallback(question_type, gold_answer, model_answer, retrieved_context, f"api_error:{exc}")
            except Exception as exc:
                return self._fallback(question_type, gold_answer, model_answer, retrieved_context, f"judge_error:{exc}")

        if response is None:
            return self._fallback(question_type, gold_answer, model_answer, retrieved_context, "judge_retry_exhausted")

        content = response.choices[0].message.content if response.choices else ""
        try:
            payload = json.loads(content or "{}")
        except json.JSONDecodeError:
            return self._fallback(question_type, gold_answer, model_answer, retrieved_context, "invalid_json_from_judge")

        semantic = _clamp_01(payload.get("semantic_correctness", 0.0))
        groundedness = _clamp_01(payload.get("groundedness", 0.0))
        explanation = str(payload.get("explanation", ""))
        return {
            "semantic_correctness": semantic,
            "groundedness": groundedness,
            "explanation": explanation,
        }
