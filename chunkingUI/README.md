# RAG Run Explorer (Web UI)

## Run locally
From the repository root:

```bash
python3 -m http.server 8000
```

Open:

- `http://localhost:8000/chunkingUI/`

## Generate run files first
From the repository root:

```bash
python3 runner.py --pipeline P0
python3 runner.py --pipeline P1
```

## Load data
- **Load JSONL file**: choose any `outputs/P0/run_*.jsonl` or `outputs/P1/run_*.jsonl` from your machine.
- **Load Path**: enter a repo-relative HTTP path like `/outputs/P0/run_20260219_171613.jsonl` or `/outputs/P1/run_20260221_153458.jsonl`.
- **Load latest pipeline output**: pick `P0` or `P1` in the selector, then click **Load Latest**.

## What it shows
- Run-level metrics (records, papers, latency, cost, scores, tokens)
- Filterable question table
- Drilldown panel with question/reference/model answer and top retrieved chunks
