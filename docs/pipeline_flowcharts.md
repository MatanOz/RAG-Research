# Pipeline Flowcharts

These diagrams use Mermaid and are organized around four blocks:

- `Document block`
- `Retrieval block`
- `Answering block`
- `Reasoning & orchestration block`

## Shared Block View

```mermaid
flowchart LR
    subgraph D["Document Block"]
        d1[Paper PDF]
        d2[Chunking and metadata]
        d3[Chunk embeddings]
        d4[(Per-paper Chroma collection)]
        d1 --> d2 --> d3 --> d4
    end

    subgraph R["Retrieval Block"]
        r1[Question]
        r2[Retrieve top chunks]
        r3[Retrieved chunk set]
        r1 --> r2 --> r3
    end

    subgraph A["Answering Block"]
        a1[Build context from chunks]
        a2[Generate answer or draft]
        a3[Answer payload]
        a1 --> a2 --> a3
    end

    subgraph O["Reasoning and Orchestration Block"]
        o1[LangGraph controller]
        o2[Select next node]
        o3[Optional critique or loop]
        o1 --> o2 --> o3
    end

    d4 -. source .-> r2
    r3 --> a1
    o2 -. controls .-> r2
    o3 -. controls .-> a2
```

## Document Block Variants

### P0 Document Block

```mermaid
flowchart LR
    p0d1[PDF pages via fitz]
    p0d2[Concatenate page text]
    p0d3[Fixed-char chunking]
    p0d4[Chunk overlap]
    p0d5[Embed chunks]
    p0d6[(Chroma collection)]

    p0d1 --> p0d2 --> p0d3 --> p0d4 --> p0d5 --> p0d6
```

### P1 to P5 Document Block

```mermaid
flowchart LR
    sd1[PDF]
    sd2[Markdown extraction via pymupdf4llm]
    sd3[Trim references tail]
    sd4[Detect section headers]
    sd5[Semantic section chunking]
    sd6[Add heuristic metadata]
    sd7[Embed chunks]
    sd8[(Chroma collection)]

    sd1 --> sd2 --> sd3 --> sd4 --> sd5 --> sd6 --> sd7 --> sd8
```

## Pipeline Versions

### P0 Baseline

```mermaid
flowchart LR
    subgraph D["Document Block"]
        p0d[P0 fixed-char Chroma collection]
    end

    subgraph R["Retrieval Block"]
        p0r1[Question]
        p0r2[Embed question]
        p0r3[Dense Chroma search]
        p0r4[Top-k chunks]
        p0r1 --> p0r2 --> p0r3 --> p0r4
    end

    subgraph A["Answering Block"]
        p0a1[Concatenate chunk text]
        p0a2[One-shot LLM answer]
        p0a3[Final answer]
        p0a1 --> p0a2 --> p0a3
    end

    subgraph O["Reasoning and Orchestration Block"]
        p0o1[START]
        p0o2[retrieve_node]
        p0o3[generate_node]
        p0o4[END]
        p0o1 --> p0o2 --> p0o3 --> p0o4
    end

    p0d -. source .-> p0r3
    p0r4 --> p0a1
    p0o2 -. runs .-> p0r3
    p0o3 -. runs .-> p0a2
```

### P1 Semantic Chunking

```mermaid
flowchart LR
    subgraph D["Document Block"]
        p1d[Semantic markdown Chroma collection]
    end

    subgraph R["Retrieval Block"]
        p1r1[Question]
        p1r2[Embed question]
        p1r3[Dense Chroma search]
        p1r4[Top-k semantic chunks]
        p1r1 --> p1r2 --> p1r3 --> p1r4
    end

    subgraph A["Answering Block"]
        p1a1[Concatenate chunk text]
        p1a2[One-shot LLM answer]
        p1a3[Final answer]
        p1a1 --> p1a2 --> p1a3
    end

    subgraph O["Reasoning and Orchestration Block"]
        p1o1[START]
        p1o2[retrieve_node]
        p1o3[generate_node]
        p1o4[END]
        p1o1 --> p1o2 --> p1o3 --> p1o4
    end

    p1d -. source .-> p1r3
    p1r4 --> p1a1
    p1o2 -. runs .-> p1r3
    p1o3 -. runs .-> p1a2
```

### P2 Hybrid Retrieval

```mermaid
flowchart LR
    subgraph D["Document Block"]
        p2d[Semantic markdown Chroma collection with metadata]
    end

    subgraph R["Retrieval Block"]
        p2q[Question]
        p2e[Embed question]
        p2dense[Dense Chroma candidates]
        p2tok[Tokenize question]
        p2bm25[BM25 sparse candidates]
        p2fuse[RRF fusion]
        p2boost[Metadata boost rules]
        p2top[Top-k fused chunks]

        p2q --> p2e --> p2dense
        p2q --> p2tok --> p2bm25
        p2dense --> p2fuse
        p2bm25 --> p2fuse --> p2boost --> p2top
    end

    subgraph A["Answering Block"]
        p2a1[Concatenate chunk text]
        p2a2[One-shot LLM answer]
        p2a3[Final answer]
        p2a1 --> p2a2 --> p2a3
    end

    subgraph O["Reasoning and Orchestration Block"]
        p2o1[START]
        p2o2[retrieve_node]
        p2o3[generate_node]
        p2o4[END]
        p2o1 --> p2o2 --> p2o3 --> p2o4
    end

    p2d -. dense source .-> p2dense
    p2d -. sparse corpus .-> p2bm25
    p2top --> p2a1
    p2o2 -. runs .-> p2fuse
    p2o3 -. runs .-> p2a2
```

### P2 Improved Adaptive Hybrid

```mermaid
flowchart LR
    subgraph D["Document Block"]
        p2id[Semantic markdown Chroma collection with metadata]
    end

    subgraph R["Retrieval Block"]
        p2iq[Question]
        p2icheck{Boost rules exist?}
        p2ie[Embed question]
        p2idense[Dense candidates]
        p2ibm25[BM25 candidates]
        p2ifuse[Weighted RRF]
        p2iboost[Metadata boost]
        p2itop[Top-k chunks]

        p2iq --> p2icheck
        p2iq --> p2ie --> p2idense
        p2icheck -- yes --> p2ibm25
        p2icheck -- no --> p2iboost
        p2idense --> p2ifuse
        p2ibm25 --> p2ifuse --> p2iboost --> p2itop
        p2idense --> p2iboost
    end

    subgraph A["Answering Block"]
        p2ia1[Concatenate chunk text]
        p2ia2[One-shot LLM answer]
        p2ia3[Final answer]
        p2ia1 --> p2ia2 --> p2ia3
    end

    subgraph O["Reasoning and Orchestration Block"]
        p2io1[START]
        p2io2[retrieve_node]
        p2io3[generate_node]
        p2io4[END]
        p2io1 --> p2io2 --> p2io3 --> p2io4
    end

    p2id -. dense source .-> p2idense
    p2id -. sparse corpus .-> p2ibm25
    p2itop --> p2ia1
    p2io2 -. runs .-> p2itop
    p2io3 -. runs .-> p2ia2
```

### P3 Adaptive Multi-Query Structured QA

```mermaid
flowchart LR
    subgraph D["Document Block"]
        p3d[Semantic markdown Chroma collection with metadata]
    end

    subgraph R["Retrieval Block"]
        p3q[Question]
        p3exp[Query expansion map]
        p3queries[Original plus up to 2 expanded queries]
        p3retr[Adaptive hybrid retrieval per query]
        p3merge[Merge by best chunk score]
        p3top[Top-5 chunks]

        p3q --> p3exp --> p3queries --> p3retr --> p3merge --> p3top
    end

    subgraph A["Answering Block"]
        p3a1[Build retrieved context]
        p3a2[Structured LLM generation]
        p3a3[Answer]
        p3a4[Reasoning]
        p3a5[Evidence quotes]
        p3a1 --> p3a2
        p3a2 --> p3a3
        p3a2 --> p3a4
        p3a2 --> p3a5
    end

    subgraph O["Reasoning and Orchestration Block"]
        p3o1[START]
        p3o2[retrieve_node]
        p3o3[generate_node]
        p3o4[END]
        p3o1 --> p3o2 --> p3o3 --> p3o4
    end

    p3d -. source .-> p3retr
    p3top --> p3a1
    p3o2 -. runs .-> p3retr
    p3o3 -. runs .-> p3a2
```

### P4 Draft Plus Conditional Critic

```mermaid
flowchart LR
    subgraph D["Document Block"]
        p4d[Semantic markdown Chroma collection with metadata]
    end

    subgraph R["Retrieval Block"]
        p4r1[Question]
        p4r2[P3 retrieval stack]
        p4r3[Top-5 chunks]
        p4r1 --> p4r2 --> p4r3
    end

    subgraph A["Answering Block"]
        p4a1[Draft generator]
        p4a2[Draft answer]
        p4a3[Reasoning]
        p4a4[Quotes]
        p4a5[Critic editor]
        p4a6[Final answer]
        p4a1 --> p4a2
        p4a1 --> p4a3
        p4a1 --> p4a4
        p4a5 --> p4a6
    end

    subgraph O["Reasoning and Orchestration Block"]
        p4o1[START]
        p4o2[retrieve_node]
        p4o3[generate_draft_node]
        p4o4{Question requires critique?}
        p4o5[bypass_node]
        p4o6[critique_node]
        p4o7[END]

        p4o1 --> p4o2 --> p4o3 --> p4o4
        p4o4 -- no --> p4o5 --> p4o7
        p4o4 -- yes --> p4o6 --> p4o7
    end

    p4d -. source .-> p4r2
    p4r3 --> p4a1
    p4o3 -. runs .-> p4a1
    p4o5 -. promote draft .-> p4a2
    p4o6 -. runs .-> p4a5
```

### P5 Ver1 Autonomous Critic Loops

```mermaid
flowchart TD
    subgraph D["Document Block"]
        p5d[Semantic markdown Chroma collection with metadata]
    end

    subgraph R["Retrieval Block"]
        p5r1[Question or critic search query]
        p5r2[Adaptive hybrid retrieval]
        p5r3[Merge new chunks with prior chunks]
        p5r4[Current evidence set]
        p5r1 --> p5r2 --> p5r3 --> p5r4
    end

    subgraph A["Answering Block"]
        p5a1[Drafter]
        p5a2[Draft answer]
        p5a3[Reasoning]
        p5a4[Quotes]
        p5a1 --> p5a2
        p5a1 --> p5a3
        p5a1 --> p5a4
    end

    subgraph O["Reasoning and Orchestration Block"]
        p5o1[START]
        p5o2[retrieve_node]
        p5o3[generate_draft_node]
        p5o4[critique_node]
        p5o5{Critic status}
        p5o6[ACCEPT]
        p5o7[REVISE]
        p5o8[RE_RETRIEVE]
        p5o9[ABSTAIN]
        p5o10[END]

        p5o1 --> p5o2 --> p5o3 --> p5o4 --> p5o5
        p5o5 --> p5o6 --> p5o10
        p5o5 --> p5o7 --> p5o3
        p5o5 --> p5o8 --> p5o2
        p5o5 --> p5o9 --> p5o10
    end

    p5d -. source .-> p5r2
    p5r4 --> p5a1
    p5o2 -. runs .-> p5r2
    p5o3 -. runs .-> p5a1
    p5o4 -. may add feedback or query .-> p5o5
```

## Evolution Summary

```mermaid
flowchart LR
    e0[P0<br/>Baseline dense RAG]
    e1[P1<br/>Better document block]
    e2[P2<br/>Hybrid retrieval]
    e3[P2_imp<br/>Adaptive retrieval policy]
    e4[P3<br/>Multi-query plus structured answer]
    e5[P4<br/>Conditional critic]
    e6[P5_ver1<br/>Looping critic controller]

    e0 --> e1 --> e2 --> e3 --> e4 --> e5 --> e6
```
