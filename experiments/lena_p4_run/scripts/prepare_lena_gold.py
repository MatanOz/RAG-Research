"""Convert Lena's Excel answers matrix into local Lena P4 gold JSON."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from xml.etree import ElementTree as ET


RUN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = RUN_DIR / "data_4_lena" / "AI Project - Answers Updated.xlsx"
DEFAULT_GOLD_OUT = RUN_DIR / "data_4_lena" / "lena_gold_answers_run4.json"
DEFAULT_MANIFEST_OUT = RUN_DIR / "data_4_lena" / "lena_paper_manifest_run4.json"

XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

QUESTION_TYPE_BY_ID = {
    1: "LIST",
    2: "STRING",
    3: "STRING",
    4: "STRING",
    5: "STRING",
    6: "STRING",
    7: "STRING",
    8: "CATEGORICAL",
    9: "STRING",
    10: "LIST",
    11: "LIST",
    12: "LIST",
    13: "STRING",
    14: "LIST",
    15: "LIST",
    16: "LIST",
    17: "LIST",
    18: "LIST",
    19: "NUMERIC",
    20: "CATEGORICAL",
    21: "LIST",
    22: "STRING",
    23: "STRING",
    24: "STRING",
    25: "STRING",
    26: "LIST",
    27: "CATEGORICAL",
    28: "LIST",
    29: "NUMERIC",
    30: "STRING",
    31: "STRING",
    32: "STRING",
    33: "CATEGORICAL",
    34: "STRING",
    35: "STRING",
    36: "STRING",
    37: "STRING",
    38: "STRING",
    39: "STRING",
    40: "NUMERIC",
    41: "STRING",
    42: "NUMERIC",
    43: "STRING",
    44: "NUMERIC",
    45: "STRING",
    46: "STRING",
    47: "STRING",
    48: "STRING",
    49: "STRING",
}


def _cell_col_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref or "A")
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def _shared_string_text(shared_item: ET.Element) -> str:
    return "".join(
        text_node.text or ""
        for text_node in shared_item.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
    )


def _inline_string_text(cell: ET.Element) -> str:
    inline = cell.find("a:is", XLSX_NS)
    if inline is None:
        return ""
    return _shared_string_text(inline)


def _read_xlsx_first_sheet(path: Path) -> List[List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    with zipfile.ZipFile(path) as archive:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [_shared_string_text(item) for item in shared_root.findall("a:si", XLSX_NS)]

        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: List[List[str]] = []
        for row in sheet_root.findall(".//a:sheetData/a:row", XLSX_NS):
            by_col: Dict[int, str] = {}
            max_col = -1
            for cell in row.findall("a:c", XLSX_NS):
                col_idx = _cell_col_index(cell.attrib.get("r", "A"))
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", XLSX_NS)
                if cell_type == "s" and value_node is not None and value_node.text:
                    value = shared[int(value_node.text)]
                elif cell_type == "inlineStr":
                    value = _inline_string_text(cell)
                elif value_node is not None:
                    value = value_node.text or ""
                else:
                    value = ""
                by_col[col_idx] = value
                max_col = max(max_col, col_idx)
            rows.append([by_col.get(idx, "") for idx in range(max_col + 1)])
    return rows


def _cell(row: Sequence[str], index: int) -> str:
    if index >= len(row):
        return ""
    return str(row[index] or "")


def _clean_text(value: str, *, multiline: bool = False) -> str:
    text = value.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    if multiline:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        return "\n".join(line for line in lines if line)
    return re.sub(r"\s+", " ", text).strip()


def _question_id(raw: str) -> int | None:
    match = re.match(r"\s*(\d+)", raw or "")
    if not match:
        return None
    return int(match.group(1))


def _question_rows(rows: Sequence[Sequence[str]]) -> List[Tuple[int, int, str]]:
    starts: List[Tuple[int, int, str]] = []
    for idx, row in enumerate(rows):
        qid = _question_id(_cell(row, 0))
        question = _clean_text(_cell(row, 1))
        if qid is not None and question:
            starts.append((idx, qid, question))
    return starts


def _paper_manifest(rows: Sequence[Sequence[str]]) -> List[Dict[str, Any]]:
    if len(rows) < 3:
        raise ValueError("Expected paper names in row 1 and paper labels in row 3.")

    source_row = rows[0]
    label_row = rows[2]
    manifest: List[Dict[str, Any]] = []
    for col_idx in range(2, max(len(source_row), len(label_row))):
        label = _clean_text(_cell(label_row, col_idx))
        source_name = _clean_text(_cell(source_row, col_idx))
        if not label and not source_name:
            continue
        match = re.search(r"(\d+)", label)
        paper_id = int(match.group(1)) if match else len(manifest) + 1
        manifest.append(
            {
                "paper_id": paper_id,
                "paper_label": label or f"Paper {paper_id}",
                "source_file_name": source_name,
                "excel_column_index": col_idx + 1,
                "expected_pdf_names": [
                    f"paper_{paper_id:02d}.pdf",
                    source_name if source_name.lower().endswith(".pdf") else f"{source_name}.pdf",
                ],
            }
        )
    return manifest


def convert_lena_workbook(
    *,
    xlsx_path: Path = DEFAULT_XLSX,
    gold_out: Path = DEFAULT_GOLD_OUT,
    manifest_out: Path = DEFAULT_MANIFEST_OUT,
) -> Tuple[Path, Path]:
    rows = _read_xlsx_first_sheet(xlsx_path)
    manifest = _paper_manifest(rows)
    question_starts = _question_rows(rows)
    if not question_starts:
        raise ValueError(f"No question rows found in {xlsx_path}")

    question_bounds: List[Tuple[int, int, int, str]] = []
    for index, (row_idx, qid, question) in enumerate(question_starts):
        next_row_idx = question_starts[index + 1][0] if index + 1 < len(question_starts) else len(rows)
        question_bounds.append((row_idx, next_row_idx, qid, question))

    records: List[Dict[str, Any]] = []
    for paper in manifest:
        col_idx = int(paper["excel_column_index"]) - 1
        for start_row, end_row, qid, question in question_bounds:
            answer_lines = [
                _clean_text(_cell(rows[row_idx], col_idx), multiline=True)
                for row_idx in range(start_row, end_row)
            ]
            reference_answer = "\n".join(line for line in answer_lines if line)
            records.append(
                {
                    "paper_id": int(paper["paper_id"]),
                    "paper_label": str(paper["paper_label"]),
                    "source_file_name": str(paper["source_file_name"]),
                    "question_id": int(qid),
                    "question": question,
                    "reference_answer": reference_answer,
                    "question_type": QUESTION_TYPE_BY_ID.get(int(qid), "FREE_TEXT"),
                    "is_answerable": bool(reference_answer.strip()),
                    "evidence_para_ids": [],
                }
            )

    gold_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    gold_out.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return gold_out, manifest_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Lena run 4 gold JSON from the Excel answers matrix.")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--gold-out", type=Path, default=DEFAULT_GOLD_OUT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_path, manifest_path = convert_lena_workbook(
        xlsx_path=args.xlsx,
        gold_out=args.gold_out,
        manifest_out=args.manifest_out,
    )
    records = json.loads(gold_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Wrote {len(records)} question records for {len(manifest)} Lena papers.")
    print(f"Gold: {gold_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
