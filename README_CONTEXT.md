# Project Context: Multi-Agent RAG for Chemical Research

## Overview
[cite_start]A research project evaluating 6 RAG architectures (P0 to P5) for extracting information from Chemistry PDFs [cite: 11-14].

## Core Architecture Constraints
- [cite_start]**Modular Design:** Every block (Ingestion, Retrieval, Generation, Evaluation) must be a separate, replaceable class or module [cite: 100-101].
- [cite_start]**Data Contract:** All pipeline outputs MUST strictly follow `pipeline_output_schema_v1.json`[cite: 232].
- [cite_start]**Tech Stack:** Python 3.10+, ChromaDB (Local/Persistent), OpenAI API (gpt-4o-mini, text-embedding-3-small) [cite: 147-149].
- [cite_start]**Efficiency Tracking:** Every run must log Latency, Token Usage, and Estimated Cost [cite: 362-364].

## Pipeline Roadmap
- **P0 (Current):** Naive RAG. [cite_start]Fixed-size chunking, Dense retrieval (Top-K=3), Single-pass generation [cite: 138-144].
- [cite_start]**P1-P5:** Future increments including semantic chunking, hybrid search, and multi-agent specialization[cite: 151, 163, 198].

## Core Data Files
- **Output Schema:** Located at `specs/pipeline_output_schema_v1.json`. [cite_start]All pipeline outputs MUST validate against this.
- **Ground Truth (Gold Master):** Located at `specs/gold_master_v4_text_plus_ids.json`. Contains the questions, reference answers, and `evidence_para_ids` for evaluation.