import os

os.environ.setdefault("DATABASE_URL", "postgresql://u:p@localhost/db")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from src.rag.fusion import ReciprocalRankFusion


def test_rrf_fuses_and_orders_results():
    fusion = ReciprocalRankFusion(k=60, top_n=3)

    dense = [
        {"id": "a", "text": "Passage A"},
        {"id": "b", "text": "Passage B"},
    ]
    keyword = [
        {"id": "b", "text": "Passage B"},
        {"id": "c", "text": "Passage C"},
    ]

    # "b" appears in both result lists, so it should win after fusion.
    texts = fusion.fuse(dense, keyword)
    assert texts == ["Passage B", "Passage A", "Passage C"]


def test_rrf_respects_top_n():
    fusion = ReciprocalRankFusion(k=60, top_n=1)
    dense = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]
    assert fusion.fuse(dense, []) == ["A"]
