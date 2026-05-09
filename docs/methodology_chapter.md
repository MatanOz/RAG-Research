# Methodology Chapter Draft

## 1. Methodology Overview

This project was designed as a controlled study of retrieval-augmented generation (RAG) pipelines for question answering over chemistry papers in PDF format. The central methodological goal was not only to build a working system, but to compare several architectural variants in a way that made their differences interpretable. For that reason, the system was organized as a sequence of pipeline versions, from `P0` to `P5`, where each version introduced a limited and intentional change relative to the previous one.

To make those comparisons rigorous, the architecture was decomposed into a small set of stable functional blocks. Each pipeline variant reused the same broad structure and output contract, while changing one or more internal blocks. This made it possible to evaluate whether an improvement came from better document preparation, stronger retrieval, more structured answer generation, or more advanced orchestration logic, rather than from an uncontrolled mixture of changes.

The present chapter focuses on the methodology up to `P3`. Pipelines `P4` and `P5` are intentionally left out at this stage because they introduce critic-based post-processing and looped control logic, which deserve a separate treatment.

## 2. Block-Based Design

### 2.1 Motivation for a Block-Based Architecture

I adopted a block-based architecture in order to isolate responsibilities and make the evolution of the system measurable. In a RAG system, several stages contribute to the final answer: documents must be parsed and chunked, evidence must be retrieved, answers must be generated from retrieved evidence, and the entire process must be coordinated. If all of those responsibilities are mixed together in one monolithic pipeline, it becomes difficult to understand why one version performs better than another. A block-based design avoids that problem.

This design also supports ablation-style experimentation. Because the high-level pipeline is stable, I can change one block while keeping the others fixed, and then attribute performance changes more confidently. For example, the transition from `P0` to `P1` mainly changes the document block, while the transition from `P1` to `P2` mainly changes the retrieval block. This makes the pipeline family easier to analyze scientifically.

At a software level, the same design improves maintainability. Shared concerns such as state handling, token accounting, and output schema validation are implemented once and reused across versions. Version-specific logic then remains localized inside the relevant pipeline class or document-processing component.

### 2.2 The Four Core Blocks

The architecture is organized into four main blocks: the document block, the retrieval block, the answering block, and the reasoning and orchestration block.

| Block | Role in the system | Main methodological question |
|---|---|---|
| Document block | Converts PDFs into searchable chunks and metadata | How should scientific documents be segmented and represented? |
| Retrieval block | Selects the chunks most relevant to a question | How can relevant evidence be found reliably? |
| Answering block | Produces the final answer from retrieved evidence | How should the model express the answer? |
| Reasoning and orchestration block | Controls the order and logic of pipeline steps | How simple or agentic should the pipeline controller be? |

### 2.3 Document Block

The document block is responsible for transforming a raw paper PDF into an indexed representation that can later support retrieval. This includes text extraction, chunking, metadata generation, embedding, and storage in ChromaDB. Methodologically, this block determines what the system considers a retrieval unit. In other words, it defines the granularity and structure of the evidence space.

This is especially important for scientific documents. Chemistry papers contain section boundaries, procedures, measured values, formulas, and references, all of which can affect downstream retrieval quality. A poorly designed document block can make retrieval fail even when the relevant information exists in the paper. For that reason, the early pipeline versions devote significant attention to chunk construction.

### 2.4 Retrieval Block

The retrieval block receives a user question and returns the chunk set that will be used as evidence for answer generation. Methodologically, this block tests how best to bridge the gap between the wording of the question and the language used in the paper. Dense retrieval, sparse retrieval, rank fusion, metadata-aware scoring, and query expansion all belong to this block.

In this project, retrieval is treated as a primary research variable rather than a fixed utility layer. Several pipeline versions differ mainly in their retrieval logic, precisely because retrieval errors are one of the main failure modes in domain-specific RAG.

### 2.5 Answering Block

The answering block consumes the retrieved context and produces the answer payload. In the simplest case, this is a single free-form answer string. In later versions, the answering block becomes more structured and may also produce supporting reasoning text and evidence quotes.

The key methodological question here is not only whether the answer is correct, but also whether the answer format is useful for evaluation and analysis. A structured answer is easier to inspect, easier to compare across pipelines, and better aligned with evidence-grounded QA.

### 2.6 Reasoning and Orchestration Block

The reasoning and orchestration block determines how the pipeline moves from one stage to the next. In the earlier pipelines covered in this chapter, the orchestration logic is intentionally simple: the graph begins, retrieves evidence, generates an answer, and ends. The reason for keeping this block simple at first is methodological. If the controller itself becomes too complex too early, it becomes harder to attribute improvements to the document, retrieval, or answering blocks.

Even in these early pipelines, however, orchestration is still explicit. The system uses a shared `AgentState` and a graph-based execution model so that later versions can evolve naturally into more complex controllers without needing a separate rewrite of the overall framework.

### 2.7 Why This Design Matters for Evaluation

The block-based design was also chosen because all pipeline outputs must follow the same structured schema. That shared schema means that retrieval outputs, generated answers, token usage, latency, and estimated cost can all be compared in a common evaluation framework. In effect, the block-based design supports not only implementation clarity, but also evaluation fairness.

## 3. YAML Configuration Architecture

### 3.1 Why I Chose YAML

I chose a YAML-based configuration architecture so that experimental conditions would be separated from implementation logic. In this project, each pipeline is not just a piece of code; it is an experimental setup with concrete parameters such as chunk size, overlap, embedding model, top-`k`, generation model, question subset, and output path. Hardcoding those parameters inside Python classes would have made the experiments harder to reproduce and harder to compare.

YAML provided a clean solution for that problem. It is human-readable, easy to version-control, and explicit enough to serve as an experimental record. Each run can be traced back to a specific configuration file, which is especially useful when comparing multiple architectures or rerunning a subset of questions under the same settings.

### 3.2 Separation Between Logic and Experimental Settings

The configuration architecture separates the system into two layers. The Python code defines what a pipeline can do, while the YAML files define how a specific experiment should run. This separation was intentional for three reasons.

First, it improves reproducibility. A configuration file captures the exact settings that produced a run, including retrieval parameters, model names, and question selection rules. Second, it supports fair comparison, because two pipelines can differ in a controlled way rather than through hidden parameter drift. Third, it accelerates iteration, because I can test a new condition by editing a configuration file instead of rewriting code.

### 3.3 Structure of the YAML Files

Each pipeline has a dedicated YAML file under `configs/`, for example `p0_baseline.yaml`, `p1_semantic.yaml`, `p2_hybrid.yaml`, `p2_imp_hybrid.yaml`, and `p3_adaptive_structured.yaml`. Although the values differ, the files share the same logical structure.

| YAML section | Purpose |
|---|---|
| `pipeline_version` | Selects the pipeline implementation from the registry |
| `retrieval_params` | Controls chunking, embedding model, top-`k`, and embedding batch size |
| `llm_params` | Controls answer-generation model and generation settings |
| `paths` | Defines data, Chroma, and output locations |
| `run_control` | Restricts papers and questions for controlled experiments |
| `logging` | Controls runtime logging and progress reporting |

This structure was designed to reflect the experimental workflow. Retrieval settings belong together because they define the evidence-selection regime. Generation settings belong together because they define the answering regime. Run control is separated because selecting which papers and questions to run is an experimental concern rather than a modeling concern.

### 3.4 Why YAML Was Important Methodologically

The YAML layer is not merely a convenience feature; it is part of the methodology. It allowed me to compare pipeline variants while keeping the surrounding environment stable. For example, if I wanted to test whether semantic chunking improved performance, I could keep the controller and answer generation logic constant while only changing the configuration associated with chunking and embeddings. Similarly, if I wanted to compare `P2` and `P2_imp`, I could preserve the same input corpus and answer model while changing only the retrieval strategy.

Another reason for using YAML was to keep the project extensible. As the architecture evolved, I wanted new pipelines to be plug-in variants rather than ad hoc scripts. Registering a new pipeline and pairing it with a configuration file made the project easier to scale without losing experimental discipline.

### 3.5 Data Contract and Comparability

The YAML architecture works together with a fixed output schema. Regardless of which pipeline version is executed, the final output record follows the same contract. This was important because a methodology chapter should describe not only how a system runs, but also how results become comparable across versions. By standardizing configuration at the input side and schema at the output side, the project makes cross-pipeline evaluation more defensible.

## 4. Pipeline Methodology by Version

### 4.1 Versioning Strategy

The pipeline family was developed incrementally. Each version was intended to answer a specific methodological question:

- `P0`: What is the performance of a minimal dense-retrieval baseline?
- `P1`: Does improving document representation help even when retrieval logic stays simple?
- `P2`: Does hybrid retrieval outperform dense retrieval alone?
- `P2_imp`: Can hybrid retrieval be made more selective and efficient?
- `P3`: Does multi-query retrieval and structured evidence generation improve answer quality and interpretability?

This progression was deliberate. The earlier versions keep the orchestration block simple so that improvements can be attributed mainly to document preparation, retrieval quality, and answer structure.

## 5. Pipeline P0: Dense Baseline

`P0` serves as the baseline architecture against which the later variants are compared. Its purpose is not to be the strongest possible system, but to provide a clean reference point. Methodologically, a baseline is essential because it establishes the performance level that can be achieved with minimal architectural complexity.

In the document block, `P0` uses raw page text extraction followed by fixed-size character chunking with overlap. This approach is intentionally simple and domain-agnostic. It does not attempt to model section boundaries or scientific structure. The advantage of this design is that it is easy to implement, computationally cheap, and transparent. The limitation is that chunk boundaries may ignore the semantic organization of the paper, which can fragment evidence or mix unrelated content.

In the retrieval block, `P0` uses standard dense retrieval. The question is embedded once, and the vector index returns the top `k` chunks from ChromaDB. No sparse retrieval, metadata heuristics, or reranking is used. This makes `P0` a useful control condition: if a later pipeline performs better, the improvement can be interpreted relative to a simple dense-retrieval baseline rather than to an already sophisticated system.

In the answering block, `P0` uses a single LLM call over the concatenated retrieved chunks. The model is instructed to answer from the provided context and to acknowledge insufficient evidence when necessary. This choice keeps answer generation simple and allows the study to focus first on whether the right evidence was retrieved at all.

In the reasoning and orchestration block, `P0` is a straight linear graph: retrieve, generate, end. This minimal controller is important methodologically because it avoids introducing additional reasoning layers that could obscure the contribution of retrieval.

Overall, `P0` represents the simplest fully working RAG system in the study. Its main value is as a benchmark for the later versions.

## 6. Pipeline P1: Semantic Chunking

`P1` was introduced to test a focused hypothesis: document representation matters, even when retrieval and generation remain unchanged. For that reason, `P1` keeps most of the `P0` logic intact while replacing the document block with a more structured ingestion strategy.

The main change in `P1` is the use of markdown-based PDF extraction and semantic chunking. Instead of splitting raw concatenated text into fixed windows, the pipeline first converts the PDF into markdown, removes the references tail, detects section boundaries, and then creates chunks that better reflect document structure. It also attaches heuristic metadata such as whether a chunk contains units, appears to describe a procedure, or contains material-related terms.

Methodologically, this version isolates the effect of better chunk construction. The retrieval block remains dense-only, and the answering block remains a one-shot answer generator. Therefore, if `P1` improves over `P0`, the improvement can be attributed mainly to the fact that the retrieval units are now more semantically coherent. This is an important experimental step, because poor retrieval can often be caused by poor chunking rather than by weak retrieval scoring.

Another reason for introducing `P1` before hybrid retrieval was practical: domain-specific QA over chemistry papers depends heavily on local structure. Measurements, procedures, and materials often cluster within specific sections. If chunks align more closely with those structures, the retriever has a better chance of returning evidence that is both relevant and interpretable.

In summary, `P1` tests whether a better document block alone can improve the overall QA system without yet changing the retrieval algorithm itself.

## 7. Pipeline P2: Hybrid Retrieval

`P2` was designed to test whether dense retrieval alone is sufficient for scientific QA, or whether it should be complemented by sparse lexical matching. The motivation for this version comes from the observation that chemistry questions often contain exact terminology, chemical names, units, and abbreviations that may not always be captured reliably by embeddings alone.

The document block in `P2` remains the same as in `P1`. This is methodologically important, because it means the retrieval experiment is not confounded by another change in chunking strategy. The semantic chunks and heuristic metadata introduced in `P1` are reused.

The main contribution of `P2` lies in the retrieval block. For each question, the pipeline performs both dense retrieval from ChromaDB and sparse retrieval using BM25 over the chunk corpus. The two ranked lists are then fused using reciprocal rank fusion (RRF), and the fused scores are adjusted using question-specific metadata boost rules. This produces a hybrid retrieval strategy that combines semantic similarity, lexical overlap, and lightweight domain heuristics.

This design was chosen because the failure modes of dense and sparse retrieval are different. Dense retrieval is strong when the wording of the question differs from the wording of the paper but the underlying meaning is similar. Sparse retrieval is strong when exact terminology matters. In chemistry papers, both conditions appear regularly, so combining the two retrieval modes is methodologically justified.

The answering block in `P2` remains intentionally simple. The system still performs one-shot answer generation from the retrieved chunk set. This preserves the experimental focus on retrieval quality. If `P2` outperforms `P1`, the natural interpretation is that better evidence selection, rather than more sophisticated answer generation, produced the gain.

Thus, `P2` represents the first version in which retrieval is treated as a multi-signal process rather than a purely embedding-based lookup.

## 8. Pipeline P2_imp: Adaptive Hybrid Retrieval

`P2_imp` was developed as an improvement over `P2`, but not by making retrieval uniformly more complex. Instead, the goal was to make hybrid retrieval more selective and better calibrated. The methodological question behind this version is whether all questions truly need the same retrieval strategy.

In `P2`, dense and sparse retrieval are always combined. In `P2_imp`, sparse retrieval is used conditionally, based on whether question-specific metadata boost rules exist for the question. Dense retrieval remains the default foundation, and weighted reciprocal rank fusion is used so that dense retrieval receives a larger share of the final score unless sparse evidence is explicitly useful.

This change reflects an important methodological principle: more components are not always better. If a sparse retriever is invoked for every question, it may add noise in cases where semantic similarity is already sufficient. By making sparse retrieval conditional, `P2_imp` aims to retain the strengths of `P2` while reducing unnecessary complexity and potential ranking errors.

The document block and answering block remain unchanged relative to `P2`. This preserves the experimental focus on retrieval policy rather than on representation or answer generation. As a result, `P2_imp` can be interpreted as a refinement of hybrid retrieval rather than as a new architecture in every respect.

Methodologically, `P2_imp` is valuable because it shows that retrieval design should not only ask which signals to combine, but also when to combine them.

## 9. Pipeline P3: Multi-Query Retrieval and Structured Answering

`P3` extends the methodology in two directions at once: it improves the retrieval block by introducing multiple query formulations, and it improves the answering block by moving from free-form text to structured evidence-bearing outputs.

The motivation for multi-query retrieval is that a single question formulation may not be the best key for retrieving evidence from a scientific paper. The wording used by the annotator, the researcher, and the paper itself may differ. To address that, `P3` supplements the original question with up to two additional query expansions drawn from a question-expansion map. Each query is run through the adaptive retrieval logic, and the candidate chunks are merged by chunk identity using the best observed score.

This approach is methodologically important because it treats retrieval as query-sensitive rather than fixed. Instead of assuming that one surface form captures the entire information need, `P3` allows multiple paraphrases or focused phrasings to compete for evidence. This is especially useful in technical domains where the same concept may be expressed with varying terminology.

The second major contribution of `P3` is in the answering block. Rather than returning only a plain-text answer, the model must produce a structured response consisting of an answer, a reasoning field, and supporting quotes. This change has two methodological benefits. First, it makes answers easier to inspect manually because the system exposes the evidence it claims to rely on. Second, it improves evaluation readiness because later analyses can distinguish between answer content and evidence support.

The reasoning and orchestration block in `P3` remains linear. This is a deliberate choice. Although `P3` introduces richer outputs, it does not yet introduce critique or iterative control. Therefore, the effect of structured answering can still be studied without the confound of a more agentic controller.

In methodological terms, `P3` is the first pipeline in this chapter that improves both evidence acquisition and answer representation. It therefore serves as the strongest architecture in the current scope while still remaining within a single-pass orchestration regime.

## 10. Scope Boundary

This chapter stops at `P3`. Pipelines `P4` and `P5` extend the methodology into critic-based correction and looped orchestration. Those versions are architecturally important, but they belong to a later methodological layer in which the reasoning and orchestration block becomes the primary research variable. For clarity, they should be discussed separately from the document-, retrieval-, and answering-focused progression covered here.

## 11. Summary

The methodology of this project is based on three main design decisions. First, the system is decomposed into stable architectural blocks so that improvements can be studied in a controlled way. Second, experimental settings are externalized into YAML files so that runs are reproducible, comparable, and easy to iterate. Third, the pipeline family is developed incrementally, with each version testing a specific hypothesis about document representation, retrieval quality, or answer structure.

Within the scope covered here, the progression is methodologically coherent. `P0` establishes the baseline, `P1` improves document representation, `P2` introduces hybrid retrieval, `P2_imp` refines that retrieval adaptively, and `P3` adds multi-query retrieval and structured evidence generation. Together, these versions form a clear experimental ladder for studying how a chemistry-focused RAG system improves before critic-based orchestration is introduced.
