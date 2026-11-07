import pytest

from ragscope import Document, FixedSizeChunker, RecursiveChunker, SentenceChunker
from ragscope.chunking import make_chunker
from ragscope.dataset import load_corpus
from ragscope.text import sentence_spans

CORPUS = load_corpus()
ALL = [FixedSizeChunker(), RecursiveChunker(), SentenceChunker()]


@pytest.mark.parametrize("chunker", ALL, ids=lambda c: c.name)
def test_chunk_text_is_exact_slice_of_source(chunker):
    for doc in CORPUS:
        for c in chunker.chunk(doc):
            assert c.text == doc.text[c.start : c.end]
            assert c.text == c.text.strip() and c.text


@pytest.mark.parametrize("chunker", ALL, ids=lambda c: c.name)
def test_every_non_whitespace_char_is_covered(chunker):
    for doc in CORPUS:
        covered = [False] * len(doc.text)
        for c in chunker.chunk(doc):
            for i in range(c.start, c.end):
                covered[i] = True
        missing = [i for i, ch in enumerate(doc.text) if not ch.isspace() and not covered[i]]
        assert not missing, f"{chunker.name} dropped text in {doc.doc_id}"


@pytest.mark.parametrize("chunker", ALL, ids=lambda c: c.name)
def test_chunk_ids_unique_and_ordered(chunker):
    chunks = chunker.chunk_all(CORPUS)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for doc in CORPUS:
        starts = [c.start for c in chunker.chunk(doc)]
        assert starts == sorted(starts)


def test_fixed_respects_size_and_overlap():
    doc = Document("d", "abcdefghij" * 10)
    chunks = FixedSizeChunker(size=30, overlap=10).chunk(doc)
    assert [c.start for c in chunks] == [0, 20, 40, 60, 80]
    assert all(len(c.text) <= 30 for c in chunks)
    assert chunks[-1].end == 100


@pytest.mark.parametrize("kwargs", [{"size": 0}, {"size": 10, "overlap": 10}, {"size": 10, "overlap": -1}])
def test_fixed_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        FixedSizeChunker(**kwargs)


def test_recursive_respects_size_and_prefers_paragraphs():
    text = "Alpha one. Alpha two.\n\nBeta one. Beta two.\n\nGamma."
    chunks = RecursiveChunker(size=25).chunk(Document("d", text))
    assert all(len(c.text) <= 25 for c in chunks)
    assert [c.text for c in chunks] == ["Alpha one. Alpha two.", "Beta one. Beta two.", "Gamma."]


def test_recursive_hard_cuts_unbreakable_text():
    chunks = RecursiveChunker(size=10).chunk(Document("d", "x" * 35))
    assert [len(c.text) for c in chunks] == [10, 10, 10, 5]


def test_sentence_chunker_never_cuts_a_sentence():
    chunker = SentenceChunker(size=200, overlap_sentences=0)
    for doc in CORPUS:
        sents = {doc.text[s:e] for s, e in sentence_spans(doc.text)}
        for c in chunker.chunk(doc):
            pieces = {c.text[s:e] for s, e in sentence_spans(c.text)}
            assert pieces <= sents


def test_sentence_overlap_repeats_last_sentence():
    text = "One is here. Two is here. Three is here. Four is here."
    chunks = SentenceChunker(size=28, overlap_sentences=1).chunk(Document("d", text))
    assert [c.text for c in chunks] == [
        "One is here. Two is here.",
        "Two is here. Three is here.",
        "Three is here. Four is here.",
    ]


def test_sentence_oversized_sentence_is_kept_whole_and_terminates():
    text = "Short. " + "word " * 100 + "end. Tail."
    chunks = SentenceChunker(size=20, overlap_sentences=3).chunk(Document("d", text))
    assert any(len(c.text) > 20 for c in chunks)
    assert chunks[-1].text.endswith("Tail.")


def test_sentence_splitter_handles_abbreviations_and_headings():
    text = "# Title\n\nUse a map, e.g. a dict. Then stop."
    sents = [text[s:e] for s, e in sentence_spans(text)]
    assert sents == ["# Title", "Use a map, e.g. a dict.", "Then stop."]


def test_empty_document_yields_no_chunks():
    for chunker in ALL:
        assert chunker.chunk(Document("e", "   \n ")) == []


def test_make_chunker_unknown():
    with pytest.raises(ValueError):
        make_chunker("nope")
