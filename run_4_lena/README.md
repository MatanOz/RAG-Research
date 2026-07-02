# Run 4 P4 Answer Run

This folder is isolated from the previous eval data and outputs. The P4 answer run uses:

- `data_4_lena/AI Project - Answers Updated.xlsx` as the source answer matrix.
- `data_4_lena/lena_gold_answers_run4.json` as the generated local gold/questions file.
- `data_4_lena/lena_paper_manifest_run4.json` as the generated local paper manifest.
- `articles_4_lena/` for the article PDFs.
- `chroma_lena_p4/` for the run-specific Chroma cache.
- `outputs_lena_p4/` for model answer JSONL files.
- `webui/` for the fixed P4 browser UI.
- `specs_lena/` for local copies of the P4 critic, query expansion, and boost maps.

The runner does not call the evaluator. It keeps the P4 final answer, draft reasoning, evidence quotes, critic logic, retrieved chunks, token logs, and gold answer in the JSONL output.

## Prepare the gold JSON

```bash
.venv/bin/python3 run_4_lena/scripts/prepare_lena_gold.py
```

## Add PDFs

Put the new article PDFs in:

```text
run_4_lena/articles_4_lena/
```

The runner accepts either `paper_01.pdf`, `paper_02.pdf`, etc., or filenames matching the workbook's `pdf file name` row.

Check missing article files before using model/API calls:

```bash
.venv/bin/python3 run_4_lena/scripts/run_lena_p4_answers.py --check-inputs-only
```

## Run P4 answers only

```bash
.venv/bin/python3 run_4_lena/scripts/run_lena_p4_answers.py
```

The runner retries a question if structured model output hits the token limit. It first uses the configured
`max_tokens`, then retries the same draft or critic call with a compact prompt and a larger token budget. If that
still fails, the question is written to the JSONL with `generation_error.handled: true` and the run continues.

To resume a partial JSONL without paying again for completed questions:

```bash
.venv/bin/python3 run_4_lena/scripts/run_lena_p4_answers.py --resume-output run_4_lena/outputs_lena_p4/<partial-output>.jsonl
```

The output will be named like:

```text
run_4_lena/outputs_lena_p4/lena_run4_p4_answers_<timestamp>_papers33_p01-33_q1-49_nq1617.jsonl
```

## View Outputs In The Web UI

The web UI auto-loads the bundled JSONL file in `run_4_lena/webui/data/`, so the `webui/` folder can be uploaded as a static site. When served from this repository, it can also refresh and read JSONL files from `run_4_lena/outputs_lena_p4/`.

From the repository root, run:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000/run_4_lena/webui/
```

Use **Refresh Runs** after a new pipeline run finishes.
