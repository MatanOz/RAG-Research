# RAG Research

Config-driven RAG experiments for chemistry PDFs with pluggable pipelines.

## Quick Start

### 1. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -V
```

### 2. Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

### 3. Set API key

Create `.env` with:

```env
OPENAI_API_KEY=your_key_here
```

## Run Pipelines (Easy Runner)

Use `runner.py` to choose the pipeline directly:

```bash
python3 runner.py --pipeline P0
python3 runner.py --pipeline P1
```

Use an explicit config file when needed:

```bash
python3 runner.py --config configs/p1_semantic.yaml
```

## Run Pipelines (Direct Main)

```bash
python3 -m src.main --config configs/p0_baseline.yaml
python3 -m src.main --config configs/p1_semantic.yaml
```

## Choose Papers and Questions

Edit `run_control` in the selected config (`configs/p0_baseline.yaml` or `configs/p1_semantic.yaml`):

- `paper_ids`: explicit list of paper IDs to run
- `max_papers`: limit number of papers when `paper_ids` is null
- `question_ids`: explicit list of question IDs
- `max_questions_per_paper`: limit per paper when `question_ids` is null

Example:

```yaml
run_control:
  paper_ids: [1]
  max_papers: null
  question_ids: [1, 2, 3]
  max_questions_per_paper: null
```

## Outputs

Runs are written as JSONL to:

- `outputs/P0/` for Pipeline 0
- `outputs/P1/` for Pipeline 1

## Evaluation Module

Run offline evaluation on existing run JSONL files:

```bash
python -m src.eval.run_eval --config src/eval/config.yaml
```

Before running, edit `src/eval/config.yaml`:

- `paths.gold_path`: gold dataset path
- `paths.output_path`: base output location for evaluator exports
- `runs`: pipeline label to JSONL path mapping (P0/P1/etc.)
- `judge.enabled`: set `true` to use GPT judge, `false` for deterministic fallback only

Evaluator output file naming:

- By default, evaluator exports are auto-named with UTC timestamp + pipeline labels, for example:
  - `outputs/eval/ui_dashboard_data_20260222_122854_P0_Baseline_P1_Semantic.json`

Optional output override:

```bash
# exact output file
python -m src.eval.run_eval --config src/eval/config.yaml --output outputs/eval/my_eval.json

# output directory (auto-named file created inside)
python -m src.eval.run_eval --config src/eval/config.yaml --output outputs/eval
```

## Web UI

Serve the repo root:

```bash
python3 -m http.server 8000
```

Open:

- `http://localhost:8000/webui/`

WebUI usage details are in `webui/README.md`.

## Notes

- P0 uses `fixed_char` chunking.
- P1 uses `semantic_markdown` chunking.
- For best compatibility, use Python 3.11.
