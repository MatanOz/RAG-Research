"""Rule-based QA metrics for numeric and list questions."""

from __future__ import annotations

import re
from string import punctuation
from typing import Any, Dict, List, Optional, Tuple

from pint import UnitRegistry


UNIT_REGISTRY = UnitRegistry(autoconvert_offset_to_baseunit=True)
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    cleaned = lowered.translate(str.maketrans("", "", punctuation))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _split_list_items(text: str) -> List[str]:
    if not text:
        return []

    chunks = re.split(r"[\n;]+", text)
    if len(chunks) == 1:
        chunks = re.split(r",\s*(?=[A-Za-z0-9])", text)

    items: List[str] = []
    for chunk in chunks:
        chunk = re.sub(r"^\s*(?:[-*]|\d+[\.\)]|[a-zA-Z][\.\)])\s*", "", chunk.strip())
        normalized = _normalize_text(chunk)
        if normalized:
            items.append(normalized)
    return items


def list_set_f1(gold_answer: str, model_answer: str) -> float:
    gold_items = set(_split_list_items(gold_answer))
    pred_items = set(_split_list_items(model_answer))

    if not gold_items and not pred_items:
        return 1.0
    if not gold_items or not pred_items:
        return 0.0

    tp = len(gold_items & pred_items)
    precision = tp / len(pred_items) if pred_items else 0.0
    recall = tp / len(gold_items) if gold_items else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _normalize_numeric_text(text: str) -> str:
    normalized = text.replace("−", "-").replace("–", "-").replace("μ", "u").replace("µ", "u")
    normalized = normalized.replace("°C", "degC")
    return normalized


def _extract_quantity(text: str) -> Optional[Any]:
    normalized = _normalize_numeric_text(text)
    range_match = re.search(
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:to|-)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([A-Za-z%u°][A-Za-z0-9u°/%^\-]*)",
        normalized,
        flags=re.IGNORECASE,
    )
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        unit = (range_match.group(3) or "").strip()
        value = (low + high) / 2.0
        try:
            if not unit:
                return UNIT_REGISTRY.Quantity(value)
            if unit == "%":
                return UNIT_REGISTRY.Quantity(value, "percent")
            return UNIT_REGISTRY.Quantity(value, unit)
        except Exception:
            return None

    unit_match = re.search(
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([A-Za-z%u°][A-Za-z0-9u°/%^\-]*)",
        normalized,
        flags=re.IGNORECASE,
    )
    if unit_match:
        value = float(unit_match.group(1))
        unit = (unit_match.group(2) or "").strip()
        try:
            if unit == "%":
                return UNIT_REGISTRY.Quantity(value, "percent")
            return UNIT_REGISTRY.Quantity(value, unit)
        except Exception:
            return None

    number_only_match = NUMBER_RE.search(normalized)
    if number_only_match:
        try:
            return UNIT_REGISTRY.Quantity(float(number_only_match.group(0)))
        except Exception:
            return None

    return None


def numeric_score(gold_answer: str, model_answer: str, tolerance: float = 0.05) -> float:
    gold_q = _extract_quantity(gold_answer)
    pred_q = _extract_quantity(model_answer)

    if gold_q is None or pred_q is None:
        return 0.0

    try:
        if getattr(gold_q, "units", None) is not None and getattr(pred_q, "units", None) is not None:
            pred_q = pred_q.to(gold_q.units)
        gold_value = float(getattr(gold_q, "magnitude", gold_q))
        pred_value = float(getattr(pred_q, "magnitude", pred_q))
    except Exception:
        return 0.0

    if abs(gold_value) <= 1e-12:
        return 1.0 if abs(pred_value - gold_value) <= tolerance else 0.0

    rel_error = abs(pred_value - gold_value) / abs(gold_value)
    return 1.0 if rel_error <= tolerance else 0.0


def compute_rule_based_score(
    question_type: str,
    gold_answer: str,
    model_answer: str,
    numeric_tolerance: float,
) -> Tuple[Optional[float], Dict[str, Any]]:
    qtype = (question_type or "").upper()
    if qtype == "NUMERIC":
        score = numeric_score(gold_answer, model_answer, tolerance=numeric_tolerance)
        return score, {"metric": "numeric_tolerance", "tolerance": numeric_tolerance}
    if qtype == "LIST":
        score = list_set_f1(gold_answer, model_answer)
        return score, {"metric": "set_f1"}
    return None, {"metric": "not_applicable"}
