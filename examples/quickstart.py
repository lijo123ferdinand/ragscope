"""Build a pipeline over the bundled corpus, ask a question, trace the citations.

    python examples/quickstart.py
"""

from ragscope import BM25Retriever, HybridRetriever, OverlapReranker, RAGPipeline, SentenceChunker
from ragscope.dataset import load_corpus

docs = load_corpus()
pipe = RAGPipeline(
    chunker=SentenceChunker(size=400, overlap_sentences=1),
    retriever=HybridRetriever(),  # BM25 + hashing-dense, fused with RRF
    reranker=OverlapReranker(),
).index(docs)
print(f"indexed {len(docs)} documents into {len(pipe.chunks)} chunks\n")

question = "How do producers make sure all events for one customer stay in sequence?"
print("Q:", question)
for hit in pipe.retrieve(question, k=3):
    print(f"  {hit.score:.3f}  {hit.chunk.chunk_id:<16} {hit.chunk.text[:70]!r}...")

answer = pipe.answer(question, k=3)
print("\nA:", answer.text)
for c in answer.citations:
    # Offsets point into the original document, so a UI can highlight the exact span.
    src = pipe.documents[c.chunk.doc_id].text[c.chunk.start : c.chunk.end]
    assert src == c.chunk.text
    print(f"  [{c.n}] {c.chunk.doc_id} chars {c.chunk.start}-{c.chunk.end}")

# Same question, BM25 only, for contrast.
bm25 = RAGPipeline(retriever=BM25Retriever()).index(docs)
print("\nBM25-only top hit:", bm25.retrieve(question, 1)[0].chunk.chunk_id)
