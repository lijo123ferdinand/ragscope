# ragscope

**A production-shaped RAG pipeline with an evaluation harness that tells you which chunker and retriever actually work on your data.**

Chunkers (fixed, recursive, sentence-aware), BM25 written from scratch, a dense retriever with a pluggable embedder, hybrid fusion with Reciprocal Rank Fusion, a reranker interface, and answers that cite their sources down to character offsets. On top of that sits the part most RAG repos skip: Recall@k, MRR and nDCG computed against evidence spans, plus LLM-judge faithfulness and answer relevance. `ragscope eval` prints the whole comparison grid. It runs offline in seconds with no API key; the only dependency is numpy.

## Why

Most RAG systems ship with the default chunk size from a tutorial, a vector store, and no measurement. When answers are wrong, nobody can say whether the cause was chunking, retrieval, ranking or generation. ragscope keeps every stage swappable and gives each one a number:

- **Chunking is measured, not assumed.** A chunk counts as relevant only if it contains the *whole* evidence sentence. Document-level labels would score every chunker the same and hide the main failure: a fixed window that cuts the answer in half.
- **Retrieval and generation are scored separately.** Recall, MRR and nDCG show whether the right text was found. Faithfulness and relevance show what the model did with it.
- **Everything is deterministic offline.** The default `MockProvider` gets the same prompts a real LLM would and answers in the same wire format (cited text, judge JSON). The tests run the real parsing code, and CI needs no secrets.

## Quickstart

```bash
git clone https://github.com/lijo123ferdinand/ragscope && cd ragscope
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
ragscope eval
```

Real output (Apple M1, Python 3.14, bundled corpus):

```text
ragscope eval: 14 docs, 70 questions, provider=mock, embedder=hashing, chunk_size=400

config            chunks    R@1    R@5  MRR@10  nDCG@5  Faith  AnsRel  RefHit  ms/q
----------------  ------  -----  -----  ------  ------  -----  ------  ------  ----
bm25/fixed            83  0.629  0.829   0.733   0.751  1.000   0.595   0.671  0.04
bm25/recursive        84  0.771  0.943   0.840   0.866  1.000   0.620   0.686  0.05
bm25/sentence        117  0.757  0.943   0.826   0.837  1.000   0.617   0.686  0.05
dense/fixed           83  0.529  0.843   0.678   0.714  1.000   0.586   0.657  0.29
dense/recursive       84  0.743  0.929   0.824   0.846  1.000   0.624   0.700  0.19
dense/sentence       117  0.757  0.914   0.834   0.830  1.000   0.631   0.686  0.24
hybrid/fixed          83  0.614  0.857   0.727   0.758  1.000   0.599   0.671  0.35
hybrid/recursive      84  0.771  0.943   0.843   0.866  1.000   0.621   0.671  0.32
hybrid/sentence      117  0.800  0.914   0.855   0.844  1.000   0.623   0.700  0.61

best by MRR: hybrid/sentence (MRR 0.855, R@5 0.914)
```

Ask a single question and trace each citation back to its source:

```text
$ ragscope ask "What does setting acks to all guarantee for a producer?"
A producer configured with acks set to all receives an acknowledgment only after every in-sync replica has the record, which together with a minimum in-sync replica setting protects against data loss when a broker fails [1][2].

[1] kafka_log chars 1236-1456
[2] kafka_log chars 1061-1456
```

From Python (`examples/quickstart.py`):

```python
from ragscope import HybridRetriever, OverlapReranker, RAGPipeline, SentenceChunker
from ragscope.dataset import load_corpus

pipe = RAGPipeline(
    chunker=SentenceChunker(size=400, overlap_sentences=1),
    retriever=HybridRetriever(),          # BM25 + dense, fused with RRF
    reranker=OverlapReranker(),
).index(load_corpus())

answer = pipe.answer("How do producers make sure all events for one customer stay in sequence?")
print(answer.text)                         # "... land in the same partition and stay in order [1][2]."
for c in answer.citations:
    print(c.n, c.chunk.doc_id, c.chunk.start, c.chunk.end)   # exact span in the source document
```

Other useful commands:

```bash
ragscope eval --rerankers none,overlap --format markdown   # add the reranker arm, emit a markdown table
ragscope eval --tag paraphrase                             # only the hard, vocabulary-mismatch questions
ragscope eval --corpus examples/custom_dataset/docs --qa examples/custom_dataset/qa.jsonl
ragscope eval --provider anthropic                         # real LLM generation + judging (needs ANTHROPIC_API_KEY)
ragscope eval --provider ollama --model llama3.1           # any OpenAI-compatible server
ragscope chunks raft --chunker fixed                       # see exactly how a document was cut
ragscope eval --out results.json                           # per-question rows, for error analysis
```

## Architecture

```text
 Documents ──► Chunker ──────────────► Chunks (doc_id, text, start, end)
               fixed | recursive |            │
               sentence                       ▼
                                  ┌─────── Retriever.index() ────────┐
                                  │                                  │
                         BM25Index (inverted          DenseRetriever: Embedder.fit/embed
                         index, Okapi BM25)           (hashing TF-IDF | sentence-transformers)
                                  │                                  │
 query ─► Retriever.search() ─────┴──── HybridRetriever: RRF ────────┘
                 │  (over-fetch `candidates` when a reranker is set)
                 ▼
            Reranker (identity | overlap | cross-encoder)
                 │ top-k ScoredChunks
                 ▼
     generate_answer(provider) ──► Answer(text, citations→chunk spans, invalid_citations)
                 │
                 ▼
     judge: faithfulness (claims vs context), answer relevance   ◄── LLMProvider
                 │                                                    mock | anthropic | openai/ollama
                 ▼
     evaluation: Recall@1/k, MRR@10, nDCG@k, Faith, AnsRel, RefHit ──► table / markdown / JSON
```

| Module | What it does |
|---|---|
| `chunking.py` | Three chunkers. Each returns character spans, so chunk text is always an exact slice of the source |
| `bm25.py` | Okapi BM25 over an inverted index, scored term at a time; Lucene-style non-negative IDF |
| `embedders.py` | `HashingEmbedder` (signed feature hashing of words + char 3-5-grams, sublinear TF, IDF) and an optional sentence-transformers wrapper |
| `retrievers.py` | BM25, exact-cosine dense, and hybrid (RRF over both, with configurable depth and weights) |
| `fusion.py` | Reciprocal Rank Fusion (Cormack et al., 2009) |
| `rerank.py` | `Reranker` interface, identity control arm, a transparent overlap reranker, an optional cross-encoder |
| `llm.py` | `LLMProvider` protocol: `MockProvider`, `AnthropicProvider`, `OpenAICompatProvider` (stdlib urllib) |
| `generate.py` / `judge.py` | Cited answers with citation validation; faithfulness + relevance judges with tolerant JSON parsing |
| `metrics.py` / `evaluation.py` | Pure ranking metrics; the grid runner and table formatter |

## Design decisions and trade-offs

- **Spans, not strings.** Chunks store `(start, end)` offsets into the document. A citation can be highlighted in a UI and audited. The tests assert `chunk.text == doc.text[start:end]` and that no chunker drops text.
- **RRF instead of a weighted score blend.** BM25 scores have no upper bound and depend on the query. Cosine scores sit in [-1, 1]. Adding them needs per-corpus normalization and a tuned weight, and the blend breaks when either distribution shifts. RRF uses only ranks. Each side contributes `depth=50` candidates, so a chunk ranked 12th by one retriever and 2nd by the other can still win.
- **BM25 IDF is `ln(1 + (N-df+0.5)/(df+0.5))`.** The classic Robertson-Sparck Jones form goes negative for terms in more than half the corpus, which is common in small corpora. BM25 deliberately does not stem, so the benchmark shows the lexical-mismatch problem.
- **The offline embedder is a different representation, not fake semantics.** Character n-grams match `replicated` to `replication`, which whole-word BM25 cannot, so hybrid fusion gets a real complementary signal with no model download. Hashing uses `crc32` because Python's `hash()` is salted per process. Signed hashing makes collisions cancel out on average.
- **Exact dense search.** Below roughly 1e5 chunks, a brute-force matrix product has perfect recall and takes milliseconds. An ANN index would add a dependency and an accuracy knob for no gain at that size.
- **Relevance is per evidence span.** Recall@k counts how many evidence spans are covered, so overlapping chunkers that repeat a sentence get no double credit. The nDCG ideal counts every relevant chunk in the index.
- **Refusals are excluded from faithfulness, not scored.** Scoring a refusal 1.0 would reward refusing everything. Scoring it 0.0 would punish honesty. Judge output that cannot be parsed is counted as a judge failure, never as 0.
- **Citations are checked.** A marker that points at a source that was never provided goes into `invalid_citations`, because hallucinated citations make wrong answers look sourced.
- **The mock answers the real prompts.** The system prompt carries a `ragscope-task:` line that the mock routes on. The mock's answers are extractive, with IDF-weighted sentence selection, deduplication of sentences repeated by overlapping chunks, and merged citations like `[1][2]`.

## Benchmark results (measured)

These numbers come from the bundled corpus: 14 hand-written technical documents (Raft, LSM trees, B-trees, Bloom filters, MVCC, CRDTs, TCP congestion control, Kafka, ...). The QA set has 70 questions. 50 are tagged `lexical` (the question reuses the source's wording) and 20 are tagged `paraphrase` (deliberate vocabulary mismatch, for example "What happens to the sending rate when an RTO fires?"). Chunk size is 400 characters, k=5, provider=mock, embedder=hashing. Run `ragscope eval --rerankers none,overlap --format markdown` to reproduce. All metrics are deterministic. `ms/q` is retrieval wall-clock time on an Apple M1 and varies from run to run.

**All 70 questions:**

| config | chunks | R@1 | R@5 | MRR@10 | nDCG@5 | Faith | AnsRel | RefHit | ms/q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25/fixed | 83 | 0.629 | 0.829 | 0.733 | 0.751 | 1.000 | 0.595 | 0.671 | 0.04 |
| bm25/recursive | 84 | 0.771 | 0.943 | 0.840 | 0.866 | 1.000 | 0.620 | 0.686 | 0.04 |
| bm25/sentence | 117 | 0.757 | 0.943 | 0.826 | 0.837 | 1.000 | 0.617 | 0.686 | 0.05 |
| dense/fixed | 83 | 0.529 | 0.843 | 0.678 | 0.714 | 1.000 | 0.586 | 0.657 | 0.39 |
| dense/recursive | 84 | 0.743 | 0.929 | 0.824 | 0.846 | 1.000 | 0.624 | 0.700 | 0.83 |
| dense/sentence | 117 | 0.757 | 0.914 | 0.834 | 0.830 | 1.000 | 0.631 | 0.686 | 0.30 |
| hybrid/fixed | 83 | 0.614 | 0.857 | 0.727 | 0.758 | 1.000 | 0.599 | 0.671 | 0.48 |
| hybrid/recursive | 84 | 0.771 | 0.943 | 0.843 | 0.866 | 1.000 | 0.621 | 0.671 | 0.40 |
| hybrid/sentence | 117 | 0.800 | 0.914 | 0.855 | 0.844 | 1.000 | 0.623 | 0.700 | 0.79 |
| bm25/fixed+overlap | 83 | 0.700 | 0.857 | 0.777 | 0.794 | 1.000 | 0.608 | 0.700 | 0.74 |
| bm25/recursive+overlap | 84 | 0.800 | 0.929 | 0.857 | 0.874 | 1.000 | 0.634 | 0.686 | 0.46 |
| bm25/sentence+overlap | 117 | 0.800 | 0.929 | 0.857 | 0.849 | 1.000 | 0.629 | 0.700 | 0.72 |
| dense/fixed+overlap | 83 | 0.657 | 0.857 | 0.760 | 0.779 | 1.000 | 0.609 | 0.686 | 1.54 |
| dense/recursive+overlap | 84 | 0.786 | 0.929 | 0.851 | 0.868 | 1.000 | 0.638 | 0.686 | 1.88 |
| dense/sentence+overlap | 117 | 0.771 | 0.929 | 0.840 | 0.839 | 1.000 | 0.629 | 0.686 | 1.03 |
| hybrid/fixed+overlap | 83 | 0.700 | 0.857 | 0.776 | 0.794 | 1.000 | 0.612 | 0.700 | 1.00 |
| hybrid/recursive+overlap | 84 | 0.800 | 0.929 | 0.858 | 0.874 | 1.000 | 0.634 | 0.686 | 0.96 |
| hybrid/sentence+overlap | 117 | 0.786 | 0.929 | 0.849 | 0.844 | 1.000 | 0.629 | 0.700 | 1.15 |

**Paraphrase split only (20 questions), `--tag paraphrase`:**

| config | chunks | R@1 | R@5 | MRR@10 | nDCG@5 | Faith | AnsRel | RefHit | ms/q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25/fixed | 83 | 0.350 | 0.650 | 0.499 | 0.533 | 1.000 | 0.427 | 0.300 | 0.06 |
| bm25/recursive | 84 | 0.550 | 0.850 | 0.672 | 0.717 | 1.000 | 0.470 | 0.400 | 0.03 |
| bm25/sentence | 117 | 0.450 | 0.850 | 0.603 | 0.614 | 1.000 | 0.440 | 0.350 | 0.05 |
| dense/fixed | 83 | 0.300 | 0.650 | 0.492 | 0.521 | 1.000 | 0.409 | 0.350 | 0.16 |
| dense/recursive | 84 | 0.500 | 0.800 | 0.642 | 0.666 | 1.000 | 0.472 | 0.400 | 0.14 |
| dense/sentence | 117 | 0.400 | 0.800 | 0.595 | 0.609 | 1.000 | 0.481 | 0.300 | 0.53 |
| hybrid/fixed | 83 | 0.350 | 0.700 | 0.504 | 0.554 | 1.000 | 0.412 | 0.350 | 0.41 |
| hybrid/recursive | 84 | 0.550 | 0.850 | 0.684 | 0.717 | 1.000 | 0.473 | 0.350 | 0.27 |
| hybrid/sentence | 117 | 0.500 | 0.800 | 0.644 | 0.627 | 1.000 | 0.452 | 0.350 | 2.37 |

What the numbers say:

1. **Chunking matters more than the choice of retriever.** Fixed-size windows cost about 0.10 MRR with every retriever (BM25: 0.733 fixed vs 0.840 recursive; dense: 0.678 vs 0.824). The window cuts evidence sentences in two, and overlap only partly fixes that.
2. **Hybrid gives the best top-1 retrieval when the two signals disagree.** With sentence chunks, hybrid reaches R@1 0.800, against 0.757 for BM25 and 0.757 for dense on their own. On the lexical split, hybrid/sentence has the best MRR in the grid at 0.940 (BM25 0.916, dense 0.930). With recursive chunks, hybrid equals BM25 and adds nothing.
3. **The offline embedder does not solve paraphrase.** On the paraphrase split, MRR drops from about 0.9 to 0.5-0.68 for every configuration, and hashing-dense comes out below BM25 (0.642 vs 0.672 with recursive chunks). Character n-grams help with word forms but not with synonyms. Questions like these need a semantic embedder (`--embedder st`), which was **not** measured here.
4. **The overlap reranker raises the top rank but can lower recall.** It adds 0.044 MRR to bm25/fixed and lifts R@1 in most configurations. Because it reorders 20 candidates, it pushes some correct chunks out of the top 5 (bm25/recursive R@5 goes from 0.943 to 0.929).
5. **Faith = 1.000 is expected with the mock.** The mock is extractive and copies source sentences, so it is faithful by construction. With the mock, AnsRel and RefHit (does the answer contain the reference answer?) track retrieval quality. Use a real provider to measure the generator itself.

Ablation from `examples/custom_components.py` (dense retrieval on all 70 questions): the hashing embedder scores MRR@10 **0.834**; with character n-grams turned off it scores **0.790**. A deliberately bad shortest-chunk-first reranker drops MRR to **0.223**, which shows the harness catches regressions.

## Using your own data

Point `--corpus` at a folder of `.md`/`.txt` files (the file stem becomes the doc_id). Point `--qa` at a JSON list or JSONL file:

```json
{"id": "c2", "question": "What is the foam on an espresso shot called?", "answer": "crema",
 "evidence": [{"doc_id": "espresso", "text": "The layer of reddish-brown foam on top of a fresh shot is called crema"}],
 "tags": ["lexical"]}
```

Each evidence string must appear verbatim in its document (whitespace and case are ignored). `ragscope eval` checks this first and refuses to run on a QA set that does not match its corpus. See `examples/custom_dataset/`.

## LLM providers

| Provider | Install | Config |
|---|---|---|
| `mock` (default) | nothing | deterministic, offline |
| `anthropic` | `pip install 'ragscope[anthropic]'` | `ANTHROPIC_API_KEY`; default model `claude-sonnet-5-5`, `--model` to override |
| `openai` / `ollama` | nothing (stdlib urllib) | `OPENAI_BASE_URL` (default `http://localhost:11434/v1`), optional `OPENAI_API_KEY` |

The OpenAI-compatible client retries 429 and 5xx responses with backoff and fails fast on other 4xx errors. Implementing `complete(system, prompt, *, max_tokens) -> str` is enough to add a backend.

## Tests

```bash
pytest -q        # 72 passed
```

The suite covers BM25 scores checked against hand computation, IDF edge cases, deterministic tie-breaks, and the RRF formula. For chunkers it checks span exactness, full text coverage, overlap behavior, oversized sentences and abbreviations. It also covers embedder determinism and normalization, metrics verified by hand, citation parsing and hallucinated-citation detection, and judge JSON parsing (code fences, garbage output, clamping). The OpenAI-compatible provider is tested against a real local HTTP server (request shape, auth header, no retry on 401); the Anthropic provider uses an injected client. Last come the pipeline, the eval grid and the CLI end to end. CI runs on Linux, macOS and Windows with Python 3.11-3.13.

## Limitations

- The bundled benchmark is small (14 documents, 70 questions) and written by the author. It is good for comparing configurations and catching regressions, not for claims about performance on other data. The R@5 differences of one or two questions are within noise.
- The default embedder is a hashing TF-IDF model, not a semantic one. Real embeddings need the `st` extra, and the results for that setup are not in this README.
- Generation and judge numbers in the table come from the heuristic mock. LLM-judge scores from a real model are not reported here.
- Everything is held in memory with exact search, and there is no persistence. That suits evaluation and corpora up to roughly 1e5 chunks; it is not a vector database.
- The sentence splitter is regex-based and English-oriented.

## Roadmap

- Contextual chunk headers (prepend the document title or section to each chunk before indexing) as a measured option
- Persisted indexes, with incremental `add` and `delete`
- Bootstrap confidence intervals and paired significance tests between configurations
- An LLM reranker on the existing `Reranker` interface, plus query rewriting (HyDE, multi-query)
- Larger public benchmarks (BEIR subsets) through the same `--corpus/--qa` loader
- Per-question diff view: which questions a configuration change fixed or broke

## License

MIT, © 2026 Lijo Ferdinand
