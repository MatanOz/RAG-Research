"""Build a standalone HTML viewer for Lena P4 answer JSONL files."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any, Dict, List


RUN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = RUN_DIR / "presentation_lena_p4"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _view_record(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    generation = record.get("generation", {}) if isinstance(record.get("generation"), dict) else {}
    retrieval = record.get("retrieval", {}) if isinstance(record.get("retrieval"), dict) else {}
    lena = record.get("lena_run4", {}) if isinstance(record.get("lena_run4"), dict) else {}
    logs = record.get("logs", {}) if isinstance(record.get("logs"), dict) else {}
    top_chunks = retrieval.get("top_chunks", []) if isinstance(retrieval.get("top_chunks"), list) else []
    return {
        "index": index,
        "run_id": record.get("run_id", ""),
        "paper_id": record.get("paper_id"),
        "paper_label": lena.get("paper_label", f"Paper {record.get('paper_id', '')}"),
        "source_file_name": lena.get("source_file_name", ""),
        "question_id": record.get("question_id"),
        "question": record.get("question", ""),
        "question_type": record.get("question_type", ""),
        "reference_answer": record.get("reference_answer", ""),
        "model_answer": generation.get("model_answer", ""),
        "reasoning": generation.get("reasoning", ""),
        "evidence_quotes": generation.get("evidence_quotes") or [],
        "critique_logic": generation.get("critique_logic", ""),
        "is_abstained": generation.get("is_abstained", False),
        "retrieved_chunks": [
            {
                "rank": chunk.get("rank"),
                "score": chunk.get("score"),
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text", ""),
            }
            for chunk in top_chunks
            if isinstance(chunk, dict)
        ],
        "tokens_input": logs.get("tokens_input", 0),
        "tokens_output": logs.get("tokens_output", 0),
        "estimated_cost_usd": logs.get("estimated_cost_usd", 0),
    }


def _html(records_json: str, title: str) -> str:
    escaped_title = escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --surface: #ffffff;
      --line: #d8dde6;
      --text: #18202b;
      --muted: #586273;
      --accent: #176b87;
      --accent-2: #5b6f3a;
      --warn: #9a4f15;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--surface);
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .summary {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(420px, 0.95fr) minmax(460px, 1.05fr);
      min-height: calc(100vh - 73px);
    }}
    .list-pane {{
      border-right: 1px solid var(--line);
      background: var(--surface);
      min-width: 0;
    }}
    .detail-pane {{
      min-width: 0;
      padding: 20px 24px 32px;
      overflow: auto;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr 150px 150px;
      gap: 10px;
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }}
    input, select {{
      width: 100%;
      min-height: 36px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 7px 9px;
      font: inherit;
      font-size: 13px;
    }}
    .rows {{
      height: calc(100vh - 144px);
      overflow: auto;
    }}
    .row {{
      width: 100%;
      text-align: left;
      border: 0;
      border-bottom: 1px solid var(--line);
      background: #fff;
      padding: 12px 14px;
      cursor: pointer;
      display: grid;
      gap: 6px;
    }}
    .row:hover, .row.active {{ background: #eef6f8; }}
    .row-top {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    .question {{
      font-size: 13px;
      line-height: 1.35;
      color: var(--text);
    }}
    .answer-preview {{
      font-size: 12px;
      line-height: 1.35;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .detail-header {{
      display: grid;
      gap: 8px;
      margin-bottom: 18px;
    }}
    .detail-header h2 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
      line-height: 1.3;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      background: #fff;
    }}
    section {{
      margin: 0 0 18px;
      padding: 0 0 18px;
      border-bottom: 1px solid var(--line);
    }}
    section h3 {{
      margin: 0 0 8px;
      font-size: 13px;
      letter-spacing: 0;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .text-block {{
      white-space: pre-wrap;
      line-height: 1.5;
      font-size: 14px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .final-answer {{
      border-left: 4px solid var(--accent);
    }}
    .critic {{
      border-left: 4px solid var(--accent-2);
    }}
    .quote {{
      margin: 0 0 8px;
      padding: 10px 12px;
      border-left: 4px solid var(--warn);
      background: #fffaf4;
      line-height: 1.45;
      font-size: 13px;
    }}
    .chunk {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 10px;
      overflow: hidden;
    }}
    .chunk-head {{
      padding: 8px 10px;
      background: #f0f3f6;
      color: var(--muted);
      font-size: 12px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }}
    .chunk-text {{
      padding: 10px;
      white-space: pre-wrap;
      line-height: 1.45;
      font-size: 13px;
      max-height: 240px;
      overflow: auto;
    }}
    .empty {{
      color: var(--muted);
      padding: 24px;
    }}
    @media (max-width: 980px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .list-pane {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .rows {{ height: 42vh; }}
      .controls {{ grid-template-columns: 1fr; }}
      header {{ align-items: flex-start; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escaped_title}</h1>
    <div class="summary" id="summary"></div>
  </header>
  <main class="layout">
    <div class="list-pane">
      <div class="controls">
        <input id="search" type="search" placeholder="Search paper, question, answer, quote, critic logic">
        <select id="paperFilter"></select>
        <select id="questionFilter"></select>
      </div>
      <div class="rows" id="rows"></div>
    </div>
    <div class="detail-pane" id="detail"></div>
  </main>
  <script id="records-json" type="application/json">{records_json}</script>
  <script>
    const records = JSON.parse(document.getElementById('records-json').textContent);
    let filtered = records.slice();
    let selectedIndex = records.length ? records[0].index : null;

    const rowsEl = document.getElementById('rows');
    const detailEl = document.getElementById('detail');
    const searchEl = document.getElementById('search');
    const paperFilterEl = document.getElementById('paperFilter');
    const questionFilterEl = document.getElementById('questionFilter');
    const summaryEl = document.getElementById('summary');

    function text(value) {{
      return String(value ?? '');
    }}

    function esc(value) {{
      return text(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }}

    function fillFilters() {{
      const papers = [...new Map(records.map(r => [r.paper_id, r])).values()]
        .sort((a, b) => Number(a.paper_id) - Number(b.paper_id));
      paperFilterEl.innerHTML = '<option value="">All papers</option>' +
        papers.map(r => `<option value="${{esc(r.paper_id)}}">Paper ${{esc(r.paper_id)}}</option>`).join('');

      const qids = [...new Set(records.map(r => r.question_id))]
        .sort((a, b) => Number(a) - Number(b));
      questionFilterEl.innerHTML = '<option value="">All questions</option>' +
        qids.map(qid => `<option value="${{esc(qid)}}">Q${{esc(qid)}}</option>`).join('');
    }}

    function recordHaystack(record) {{
      return [
        record.paper_label,
        record.source_file_name,
        record.question,
        record.reference_answer,
        record.model_answer,
        record.reasoning,
        record.critique_logic,
        ...(record.evidence_quotes || []),
      ].join(' ').toLowerCase();
    }}

    function applyFilters() {{
      const query = searchEl.value.trim().toLowerCase();
      const paper = paperFilterEl.value;
      const qid = questionFilterEl.value;
      filtered = records.filter(record => {{
        if (paper && String(record.paper_id) !== paper) return false;
        if (qid && String(record.question_id) !== qid) return false;
        if (query && !recordHaystack(record).includes(query)) return false;
        return true;
      }});
      if (!filtered.some(record => record.index === selectedIndex)) {{
        selectedIndex = filtered.length ? filtered[0].index : null;
      }}
      renderRows();
      renderDetail();
      renderSummary();
    }}

    function renderSummary() {{
      const paperCount = new Set(records.map(r => r.paper_id)).size;
      const questionCount = records.length;
      const shown = filtered.length;
      summaryEl.innerHTML = `
        <span>${{shown}} shown</span>
        <span>${{questionCount}} total answers</span>
        <span>${{paperCount}} papers</span>
      `;
    }}

    function renderRows() {{
      if (!filtered.length) {{
        rowsEl.innerHTML = '<div class="empty">No records match the current filters.</div>';
        return;
      }}
      rowsEl.innerHTML = filtered.map(record => {{
        const active = record.index === selectedIndex ? ' active' : '';
        return `
          <button class="row${{active}}" data-index="${{esc(record.index)}}">
            <div class="row-top">
              <span>Paper ${{esc(record.paper_id)}} - Q${{esc(record.question_id)}} - ${{esc(record.question_type)}}</span>
              <span>${{record.is_abstained ? 'Abstained' : ''}}</span>
            </div>
            <div class="question">${{esc(record.question)}}</div>
            <div class="answer-preview">${{esc(record.model_answer || 'No answer')}}</div>
          </button>
        `;
      }}).join('');
    }}

    function quoteList(quotes) {{
      if (!quotes || !quotes.length) return '<div class="empty">No evidence quotes returned.</div>';
      return quotes.map(quote => `<blockquote class="quote">${{esc(quote)}}</blockquote>`).join('');
    }}

    function chunksList(chunks) {{
      if (!chunks || !chunks.length) return '<div class="empty">No retrieved chunks recorded.</div>';
      return chunks.map(chunk => `
        <div class="chunk">
          <div class="chunk-head">
            <span>Rank ${{esc(chunk.rank)}} - ${{esc(chunk.chunk_id || '')}}</span>
            <span>score ${{Number(chunk.score || 0).toFixed(4)}}</span>
          </div>
          <div class="chunk-text">${{esc(chunk.text || '')}}</div>
        </div>
      `).join('');
    }}

    function renderDetail() {{
      const record = records.find(item => item.index === selectedIndex);
      if (!record) {{
        detailEl.innerHTML = '<div class="empty">Select a record to inspect.</div>';
        return;
      }}
      detailEl.innerHTML = `
        <div class="detail-header">
          <h2>${{esc(record.question)}}</h2>
          <div class="meta">
            <span class="pill">Paper ${{esc(record.paper_id)}}</span>
            <span class="pill">Q${{esc(record.question_id)}}</span>
            <span class="pill">${{esc(record.question_type)}}</span>
            <span class="pill">${{esc(record.source_file_name || record.paper_label)}}</span>
            ${{record.is_abstained ? '<span class="pill">Abstained</span>' : ''}}
          </div>
        </div>
        <section>
          <h3>Final Answer</h3>
          <div class="text-block final-answer">${{esc(record.model_answer || '')}}</div>
        </section>
        <section>
          <h3>Draft Reasoning</h3>
          <div class="text-block">${{esc(record.reasoning || 'No reasoning returned.')}}</div>
        </section>
        <section>
          <h3>Evidence Quotes</h3>
          ${{quoteList(record.evidence_quotes)}}
        </section>
        <section>
          <h3>Critic Logic</h3>
          <div class="text-block critic">${{esc(record.critique_logic || 'No critic logic returned.')}}</div>
        </section>
        <section>
          <h3>Gold Answer From Lena File</h3>
          <div class="text-block">${{esc(record.reference_answer || 'No gold answer in workbook.')}}</div>
        </section>
        <section>
          <h3>Retrieved Chunks</h3>
          ${{chunksList(record.retrieved_chunks)}}
        </section>
      `;
    }}

    rowsEl.addEventListener('click', event => {{
      const row = event.target.closest('.row');
      if (!row) return;
      selectedIndex = Number(row.dataset.index);
      renderRows();
      renderDetail();
    }});
    searchEl.addEventListener('input', applyFilters);
    paperFilterEl.addEventListener('change', applyFilters);
    questionFilterEl.addEventListener('change', applyFilters);

    fillFilters();
    applyFilters();
  </script>
</body>
</html>
"""


def build_viewer(input_path: Path, output_path: Path | None = None) -> Path:
    records = [_view_record(record, index) for index, record in enumerate(_load_jsonl(input_path))]
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = output_dir / f"{input_path.stem}_viewer.html"
    records_json = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    title = f"Lena Run 4 P4 Answers - {input_path.stem}"
    output_path.write_text(_html(records_json=records_json, title=title), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a standalone Lena P4 answer viewer.")
    parser.add_argument("input", type=Path, help="Lena P4 JSONL output path.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = build_viewer(input_path=args.input, output_path=args.output)
    print(f"Wrote Lena P4 viewer to: {output_path}")


if __name__ == "__main__":
    main()
