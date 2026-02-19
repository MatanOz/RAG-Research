# RAG Run Explorer (Web UI)

## Run locally
From the repository root:

```bash
python3 -m http.server 8000
```

Open:

- `http://localhost:8000/webui/`

## Load data
- **Load JSONL file**: choose any `outputs/P0/run_*.jsonl` from your machine.
- **Load Path**: enter a repo-relative HTTP path like `/outputs/P0/run_20260219_171613.jsonl`.
- **Load Latest From outputs/P0**: auto-detects newest `run_*.jsonl` when served from repo root.

## What it shows
- Run-level metrics (records, papers, latency, cost, scores, tokens)
- Filterable question table
- Drilldown panel with question/reference/model answer and top retrieved chunks
