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
python3 runner.py --pipeline P2
python3 runner.py --pipeline P2_IMP
python3 runner.py --pipeline P3
```

Use an explicit config file when needed:

```bash
python3 runner.py --config configs/p1_semantic.yaml
```

## Run Pipelines (Direct Main)

```bash
python3 -m src.main --config configs/p0_baseline.yaml
python3 -m src.main --config configs/p1_semantic.yaml
python3 -m src.main --config configs/p2_hybrid.yaml
python3 -m src.main --config configs/p2_imp_hybrid.yaml
python3 -m src.main --config configs/p3_adaptive_structured.yaml
```

## Choose Papers and Questions

Edit `run_control` in the selected config (`configs/p0_baseline.yaml`, `configs/p1_semantic.yaml`, `configs/p2_hybrid.yaml`, `configs/p2_imp_hybrid.yaml`, or `configs/p3_adaptive_structured.yaml`):

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
- `outputs/P2/` for Pipeline 2
- `outputs/P2_imp/` for Pipeline 2 Improved
- `outputs/P3/` for Pipeline 3

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
- `evaluation.no_gold_policy`: penalizes hallucinated concrete answers when gold is empty/unmeasured and rewards explicit abstention
- `evaluation.gold_present_policy`: penalizes abstention when gold has concrete facts and applies partial-coverage caps for FREE_TEXT numeric facts

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

## Web UIs

Serve the repo root:

```bash
python3 -m http.server 8000
```

Open:

- Chunking UI (P0/P1 run explorer): `http://localhost:8000/chunkingUI/`
- Evaluation Dashboard: `http://localhost:8000/webui_eval/`

Chunking UI usage details are in `chunkingUI/README.md`.

## Quick Command Flow

```bash
# 1) run pipelines (example)
python3 runner.py --pipeline P0
python3 runner.py --pipeline P1
python3 runner.py --pipeline P2_IMP
python3 runner.py --pipeline P3

# 2) run evaluator
python -m src.eval.run_eval --config src/eval/config.yaml

# 3) launch UIs
python3 -m http.server 8000
```

## Notes

- P0 uses `fixed_char` chunking.
- P1 uses `semantic_markdown` chunking.
- P2 uses hybrid retrieval (dense + BM25 + RRF + metadata boosting).
- P2_IMP uses adaptive hybrid retrieval (conditional BM25 + weighted RRF + metadata boosting).
- P3 uses adaptive multi-query retrieval + structured evidence generation (answer/reasoning/quotes).
- For best compatibility, use Python 3.11.
