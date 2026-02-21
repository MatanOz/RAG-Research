"""Document and ChromaDB management utilities for PDF ingestion and caching."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


CHROMA_CLIENT: Any = None
EMBED_TEXTS_FN: Optional[Callable[[str, Sequence[str]], Tuple[List[List[float]], int]]] = None
LOGGER: Any = None
REBUILT_PAPERS_THIS_RUN: set = set()


def initialize_document_manager(
    chroma_client: Any,
    embed_texts_fn: Callable[[str, Sequence[str]], Tuple[List[List[float]], int]],
    logger: Any,
) -> None:
    global CHROMA_CLIENT, EMBED_TEXTS_FN, LOGGER
    CHROMA_CLIENT = chroma_client
    EMBED_TEXTS_FN = embed_texts_fn
    LOGGER = logger


def reset_rebuilt_papers() -> None:
    global REBUILT_PAPERS_THIS_RUN
    REBUILT_PAPERS_THIS_RUN = set()


def _require_context() -> Tuple[Any, Callable[[str, Sequence[str]], Tuple[List[List[float]], int]], Any, set]:
    if CHROMA_CLIENT is None or EMBED_TEXTS_FN is None or LOGGER is None:
        raise RuntimeError("Document manager context is not initialized.")
    return CHROMA_CLIENT, EMBED_TEXTS_FN, LOGGER, REBUILT_PAPERS_THIS_RUN


def _log_info(message: str, *args: Any) -> None:
    if LOGGER is not None:
        LOGGER.info(message, *args)


def stable_collection_name(paper_id: int, chunking_method: str) -> str:
    return f"paper_{paper_id:02d}_{chunking_method}"


def fingerprint_path(chroma_dir: Path, paper_id: int) -> Path:
    return chroma_dir / "meta" / f"paper_{paper_id:02d}_fingerprint.json"


def get_pdf_signature(pdf_path: Path) -> Dict[str, str]:
    try:
        digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        return {"type": "sha256", "value": digest}
    except OSError:
        stat = pdf_path.stat()
        fallback = f"{int(stat.st_mtime)}:{stat.st_size}"
        return {"type": "mtime_size", "value": fallback}


def build_fingerprint(paper_id: int, pdf_path: Path, config: Any) -> Dict[str, Any]:
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


def extract_markdown_with_pymupdf(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    _log_info("Starting extraction for %s...", pdf_path.name)
    markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
    _log_info("Extraction finished (%s chars)", len(markdown_text))
    return markdown_text


def clean_markdown_references(markdown_text: str) -> str:
    reference_pattern = re.compile(
        r"(?im)^\s*#*\s*.*(?:references|bibliography|literature cited|notes and references).*$",
        flags=re.MULTILINE,
    )
    match = reference_pattern.search(markdown_text)
    if not match:
        return markdown_text
    found_header = match.group(0).strip()
    _log_info("Cleaned references. Truncated from match: %s", found_header)
    return markdown_text[: match.start()].rstrip()


def extract_heuristic_metadata(chunk_text: str) -> Dict[str, bool]:
    has_units = bool(
        re.search(
            r"\d+\s?(mg|g|mL|L|mmol|mol|°C|K|nm|cm|%|h|min|hours|minutes)\b",
            chunk_text,
            flags=re.IGNORECASE,
        )
    )

    procedure_roots = [
        "stir",
        "heat",
        "dissolv",
        "reflux",
        "yield",
        "wash",
        "dry",
        "precipitat",
        "filter",
        "add",
        "dropwise",
    ]
    procedure_hits = 0
    for root in procedure_roots:
        if re.search(rf"\b{root}\w*\b", chunk_text, flags=re.IGNORECASE):
            procedure_hits += 1
    is_procedure = procedure_hits >= 2

    has_material_terms = bool(
        re.search(
            r"\b(THF|DMF|DCM|EtOH|MeOH|DMSO|water|ether|solvent|reagent|catalyst|acid|base|buffer)\b",
            chunk_text,
            flags=re.IGNORECASE,
        )
    )
    has_formula = bool(re.search(r"\b[A-Z][a-z]?\d+\b", chunk_text))
    has_materials = has_material_terms or has_formula

    return {
        "has_units": has_units,
        "is_procedure": is_procedure,
        "has_materials": has_materials,
    }


def _is_section_header(line: str) -> bool:
    patterns = [
        r"^#+\s+.+$",
        r"^\s*\*\*[^*]+\*\*\s*$",
        r"^\d+(\.\d+)*\s+[A-Z].+$",
        r"^[A-Z\s]{3,30}$",
    ]
    return any(re.match(pattern, line) for pattern in patterns)


def _section_name_from_header(line: str) -> str:
    stripped = line.strip()
    if re.match(r"^#+\s+.+$", stripped):
        return re.sub(r"^#+\s*", "", stripped).strip()
    if re.match(r"^\s*\*\*[^*]+\*\*\s*$", stripped):
        return stripped.strip("* ").strip()
    return stripped


def semantic_chunk_markdown(
    paper_id: int,
    markdown_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[ChunkRecord]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size for semantic_markdown chunking")

    cleaned_markdown = clean_markdown_references(markdown_text)
    lines = cleaned_markdown.splitlines()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: List[ChunkRecord] = []
    current_section = "preamble"
    current_lines: List[str] = []
    chunk_index = 0

    def flush_chunk() -> None:
        nonlocal chunk_index, current_lines
        chunk_text = "\n".join(current_lines).strip()
        current_lines = []
        if not chunk_text:
            return

        if len(chunk_text) > 1200:
            sub_chunks = splitter.split_text(chunk_text)
            _log_info(
                "Section '%s' too long (%s chars), split into %s sub-chunks using size=%s.",
                current_section,
                len(chunk_text),
                len(sub_chunks),
                chunk_size,
            )
        else:
            sub_chunks = [chunk_text]

        for sub_chunk_text in sub_chunks:
            metadata: Dict[str, Any] = {
                "paper_id": paper_id,
                "chunk_index": chunk_index,
                "section_name": current_section,
            }
            metadata.update(extract_heuristic_metadata(sub_chunk_text))
            chunks.append(
                ChunkRecord(
                    chunk_id=f"paper_{paper_id:02d}_chunk_{chunk_index:05d}",
                    text=sub_chunk_text,
                    metadata=metadata,
                )
            )
            chunk_index += 1

    for line in lines:
        stripped = line.strip()
        if _is_section_header(stripped):
            section_name = _section_name_from_header(stripped)
            if re.search(r"(references|bibliography|literature cited|notes and references)", section_name, flags=re.IGNORECASE):
                flush_chunk()
                _log_info(
                    "Reference section detected during chunking: '%s'. Stopping paper processing.",
                    stripped,
                )
                break
            flush_chunk()
            current_section = section_name
            current_lines.append(stripped)
            continue
        current_lines.append(line)

    flush_chunk()

    if chunks:
        return chunks

    empty_metadata: Dict[str, Any] = {
        "paper_id": paper_id,
        "chunk_index": 0,
        "section_name": "preamble",
    }
    empty_metadata.update(extract_heuristic_metadata(""))
    return [
        ChunkRecord(
            chunk_id=f"paper_{paper_id:02d}_chunk_00000",
            text="",
            metadata=empty_metadata,
        )
    ]


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for offset in range(0, len(items), batch_size):
        yield items[offset : offset + batch_size]


def get_or_build_chroma_collection(paper_id: int, pdf_path: Path, config: Any) -> Tuple[Any, int]:
    chroma_client, embed_texts_fn, logger, rebuilt_papers = _require_context()
    chroma_dir = Path(config.paths.chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    (chroma_dir / "meta").mkdir(parents=True, exist_ok=True)

    chunking_method = config.retrieval_params.chunking_method
    collection_name = stable_collection_name(paper_id, chunking_method)
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
        build_tokens = 0
        if existing_fp:
            build_tokens = int(existing_fp.get("paper_build_embedding_tokens", 0))
        logger.info(
            "paper=%s chroma load | collection=%s chunks=%s persist_dir=%s",
            paper_id,
            collection_name,
            count,
            chroma_dir,
        )
        return existing_collection, build_tokens

    if paper_id in rebuilt_papers:
        logger.info("paper=%s rebuild already performed in this run; reusing collection", paper_id)
        if existing_collection is None:
            existing_collection = chroma_client.get_collection(name=collection_name)
        return existing_collection, 0

    logger.info(
        "paper=%s chroma build | collection=%s reason=%s persist_dir=%s",
        paper_id,
        collection_name,
        "missing" if existing_collection is None else "fingerprint_changed",
        chroma_dir,
    )
    logger.info("paper=%s build start | method=%s", paper_id, chunking_method)

    if existing_collection is not None:
        chroma_client.delete_collection(name=collection_name)

    collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    if chunking_method == "fixed_char":
        page_texts = extract_pdf_pages(pdf_path)
        chunks = chunk_pdf_pages(
            paper_id=paper_id,
            page_texts=page_texts,
            chunk_size=config.retrieval_params.chunk_size,
            chunk_overlap=config.retrieval_params.chunk_overlap,
        )
    elif chunking_method == "semantic_markdown":
        markdown_text = extract_markdown_with_pymupdf(pdf_path)
        chunks = semantic_chunk_markdown(
            paper_id=paper_id,
            markdown_text=markdown_text,
            chunk_size=config.retrieval_params.chunk_size,
            chunk_overlap=config.retrieval_params.chunk_overlap,
        )
    else:
        raise ValueError(f"Unsupported chunking method: {chunking_method}")
    logger.info("Processing paper %s... Generated %s chunks.", paper_id, len(chunks))

    logger.info(
        "paper=%s chunking done | method=%s chunks=%s chunk_size=%s overlap=%s",
        paper_id,
        chunking_method,
        len(chunks),
        config.retrieval_params.chunk_size,
        config.retrieval_params.chunk_overlap,
    )

    batch_size = config.retrieval_params.embedding_batch_size
    batches = list(batched(chunks, batch_size))
    paper_build_tokens = 0
    for batch_index, batch in enumerate(batches, start=1):
        texts = [item.text for item in batch]
        ids = [item.chunk_id for item in batch]
        metas = [item.metadata for item in batch]
        vectors, tokens = embed_texts_fn(config.retrieval_params.embedding_model, texts)
        paper_build_tokens += tokens
        collection.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=vectors)
        logger.info(
            "paper=%s embedding progress | batch=%s/%s batch_size=%s",
            paper_id,
            batch_index,
            len(batches),
            len(batch),
        )

    current_fp["paper_build_embedding_tokens"] = int(paper_build_tokens)
    current_fp["paper_build_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_fingerprint(fp_path, current_fp)
    rebuilt_papers.add(paper_id)
    logger.info("paper=%s total chunks added to ChromaDB=%s", paper_id, len(chunks))
    logger.info("paper=%s chroma build complete | collection=%s chunks=%s", paper_id, collection_name, len(chunks))
    return collection, paper_build_tokens
