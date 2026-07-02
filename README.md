# RAG Research

**Architectural and Performance Evaluation of Multi-Agent RAG Systems for Chemical Research Information Extraction**

This repository contains a final MSc research project on retrieval-augmented
generation (RAG) for chemistry and materials-science papers. The project
evaluates a family of versioned RAG pipelines (`P0` to `P5`) for extracting
structured answers from scientific PDFs. Each pipeline changes a controlled
part of the system, such as document chunking, retrieval, answer generation,
critic-based correction, or iterative re-retrieval.

The main goal is to compare how architectural choices affect answer accuracy,
grounding, retrieval quality, hallucination behavior, latency, and cost. The
repository includes the implementation, configuration files, evaluation module,
benchmark papers, Gold Master Q&A data, final evaluation exports, and report
supporting material.

## Repository Contents

- `src/`: pipeline implementations, shared state, document processing, and evaluation code
- `configs/`: YAML experiment configurations for the pipeline variants
- `specs/`: Gold Master Q&A/evidence data, output schema, and instruction maps
- `data/`: the 10 benchmark chemistry/materials-science papers used in the final evaluation
- `docs/`: methodology draft, architecture diagrams, and report-supporting visual material
- `webui_eval/`: browser-based evaluation dashboard for inspecting pipeline outputs
- `run_4_lena/`: additional run package and scripts for a separate P4 evaluation workflow

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
python3 runner.py --pipeline P4
python3 runner.py --pipeline P5_VER1
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
python3 -m src.main --config configs/p4_agentic_corrector.yaml
python3 -m src.main --config configs/p5_ver1.yaml
```

## Choose Papers and Questions

Edit `run_control` in the selected config (`configs/p0_baseline.yaml`, `configs/p1_semantic.yaml`, `configs/p2_hybrid.yaml`, `configs/p2_imp_hybrid.yaml`, `configs/p3_adaptive_structured.yaml`, `configs/p4_agentic_corrector.yaml`, or `configs/p5_ver1.yaml`):

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
- `outputs/P4/` for Pipeline 4
- `outputs/P5_ver1/` for Pipeline 5 (version 1)

Runtime outputs are generated artifacts and are ignored by default. The final
submission keeps only the selected full evaluation exports under `outputs/eval/`.

## Final Submission Artifacts

The repository includes the durable artifacts needed to inspect the final
benchmark and reproduce the reported evaluation setup:

- source code, pipeline configs, and documentation
- the 10 benchmark papers under `data/` (`paper_01.pdf` through `paper_10.pdf`)
- the Gold Master Q&A/evidence reference under `specs/gold_master_v4_text_plus_ids.json`
- the final full evaluation exports:
  - `outputs/eval/P0_to_P2 eval full.json`
  - `outputs/eval/p3 eval full.json`
  - `outputs/eval/p4 full eval.json`
  - `outputs/eval/p5 eval full.json`

## Credits

This work was completed by **Matan Oz** as a final report for the degree of
**Master of Science in Intelligent Systems Engineering**.

Supervisor: **Dr. Yehudit Aperstein**

School of Software Engineering: Intelligent Systems

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

## Architecture Diagrams

Block-based Mermaid flowcharts for `P0` through `P5_ver1` are in:

- `docs/pipeline_flowcharts.md`
- `docs/pipeline_flowcharts.html` for a browser-rendered version

## Quick Command Flow

```bash
# 1) run pipelines (example)
python3 runner.py --pipeline P0
python3 runner.py --pipeline P1
python3 runner.py --pipeline P2_IMP
python3 runner.py --pipeline P3
python3 runner.py --pipeline P4
python3 runner.py --pipeline P5_VER1

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
- P4 uses critic-corrector generation with conditional critic routing.
- P5_VER1 uses autonomous critic-driven loops (revise / re-retrieve / accept / abstain).
- For best compatibility, use Python 3.11.
