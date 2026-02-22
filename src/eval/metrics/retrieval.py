"""Retrieval overlap metrics against gold evidence text."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def _unique_evidence_text(
    evidence_ids: Sequence[str],
    evidence_lookup: Mapping[str, str],
    fallback_text: str,
) -> str:
    chunks: List[str] = []
    seen = set()
    for para_id in evidence_ids:
        text = evidence_lookup.get(str(para_id), "")
        if text and text not in seen:
            seen.add(text)
            chunks.append(text)
    if not chunks and fallback_text:
        chunks.append(fallback_text)
    return "\n\n".join(chunks)


def token_overlap_recall(gold_text: str, retrieved_text: str) -> float:
    gold_tokens = set(_tokenize(gold_text))
    if not gold_tokens:
        return 0.0
    retrieved_tokens = set(_tokenize(retrieved_text))
    if not retrieved_tokens:
        return 0.0
    return len(gold_tokens & retrieved_tokens) / len(gold_tokens)


def _retrieved_text_from_record(record: Mapping[str, Any], k: int) -> str:
    retrieval = record.get("retrieval", {}) if isinstance(record, dict) else {}
    top_chunks = retrieval.get("top_chunks", [])
    if isinstance(top_chunks, list) and top_chunks:
        texts = [str(chunk.get("text", "")) for chunk in top_chunks[:k]]
        return "\n\n".join([txt for txt in texts if txt])
    return str(retrieval.get("retrieved_context", ""))


def evaluate_retrieval(
    record: Mapping[str, Any],
    gold_record: Mapping[str, Any],
    evidence_lookup: Mapping[str, str],
    k: int,
) -> Dict[str, Any]:
    gold_ids = record.get("gold_evidence_para_ids", []) if isinstance(record, dict) else []
    if not gold_ids:
        gold_ids = gold_record.get("evidence_para_ids", []) if isinstance(gold_record, dict) else []

    fallback_gold_text = ""
    if isinstance(gold_record, dict):
        fallback_gold_text = str(gold_record.get("gold_evidence_text", ""))

    gold_text = _unique_evidence_text(gold_ids, evidence_lookup, fallback_gold_text)
    retrieved_text = _retrieved_text_from_record(record, k=k)
    recall_at_k = token_overlap_recall(gold_text, retrieved_text)

    return {
        "recall_at_k": recall_at_k,
        "gold_evidence_text_used": gold_text,
    }

