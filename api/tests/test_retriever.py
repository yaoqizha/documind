"""
單元測試：services/retriever.py
執行：pytest tests/test_retriever.py -v
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.retriever import RetrievedChunk, _rerank


# ── RetrievedChunk dataclass 測試 ───────────────────────────

class TestRetrievedChunk:
    def test_creation(self):
        chunk = RetrievedChunk(
            content="測試內容",
            filename="doc.pdf",
            chunk_index=0,
            semantic_score=0.92,
            rerank_score=8.5,
            excerpt="測試內容",
        )
        assert chunk.content == "測試內容"
        assert chunk.rerank_score == 8.5

    def test_excerpt_is_set(self):
        long_content = "A" * 500
        chunk = RetrievedChunk(
            content=long_content,
            filename="test.txt",
            chunk_index=1,
            semantic_score=0.8,
            rerank_score=5.0,
            excerpt=long_content[:200],
        )
        assert len(chunk.excerpt) == 200


# ── _rerank 測試（mock CrossEncoder）────────────────────────

class TestRerank:
    def test_rerank_returns_top_n(self):
        candidates = [
            {"content": f"候選文件 {i}", "filename": f"doc{i}.txt",
             "chunk_index": i, "cosine_similarity": 0.9 - i * 0.1}
            for i in range(5)
        ]
        with patch("services.retriever._get_reranker") as mock_reranker_fn:
            mock_reranker = MagicMock()
            mock_reranker.predict.return_value = [0.9, 0.3, 0.7, 0.1, 0.5]
            mock_reranker_fn.return_value = mock_reranker

            results = _rerank("測試查詢", candidates, top_n=3)

        assert len(results) == 3

    def test_rerank_sorted_by_score_descending(self):
        candidates = [
            {"content": f"文件 {i}", "filename": f"doc{i}.txt",
             "chunk_index": i, "cosine_similarity": 0.8}
            for i in range(3)
        ]
        scores = [0.2, 0.9, 0.5]
        with patch("services.retriever._get_reranker") as mock_reranker_fn:
            mock_reranker = MagicMock()
            mock_reranker.predict.return_value = scores
            mock_reranker_fn.return_value = mock_reranker

            results = _rerank("查詢", candidates, top_n=3)

        assert results[0].rerank_score == 0.9
        assert results[1].rerank_score == 0.5
        assert results[2].rerank_score == 0.2

    def test_rerank_top_n_less_than_candidates(self):
        candidates = [
            {"content": f"文件 {i}", "filename": "doc.txt",
             "chunk_index": i, "cosine_similarity": 0.8}
            for i in range(10)
        ]
        with patch("services.retriever._get_reranker") as mock_reranker_fn:
            mock_reranker = MagicMock()
            mock_reranker.predict.return_value = [float(i) for i in range(10)]
            mock_reranker_fn.return_value = mock_reranker

            results = _rerank("查詢", candidates, top_n=3)

        assert len(results) == 3

    def test_rerank_empty_candidates(self):
        results = _rerank("查詢", [], top_n=3)
        assert results == []

    def test_excerpt_truncated_to_200_chars(self):
        long_content = "測試" * 200
        candidates = [
            {"content": long_content, "filename": "doc.txt",
             "chunk_index": 0, "cosine_similarity": 0.9}
        ]
        with patch("services.retriever._get_reranker") as mock_reranker_fn:
            mock_reranker = MagicMock()
            mock_reranker.predict.return_value = [0.9]
            mock_reranker_fn.return_value = mock_reranker

            results = _rerank("查詢", candidates, top_n=1)

        assert len(results[0].excerpt) == 200


# ── retrieve 整合測試（mock DB + reranker）──────────────────

@pytest.mark.asyncio
class TestRetrieve:
    async def test_retrieve_returns_empty_on_no_results(self):
        with patch("services.retriever._semantic_search", new_callable=AsyncMock) as mock_search, \
             patch("services.retriever.embed_query", new_callable=AsyncMock) as mock_embed:
            mock_search.return_value = []
            mock_embed.return_value = [0.1] * 1536
            from services.retriever import retrieve
            results = await retrieve("查詢", "tenant_a")
        assert results == []

    async def test_retrieve_calls_reranker_with_candidates(self):
        mock_candidates = [
            {"content": "相關內容", "filename": "test.pdf",
             "chunk_index": 0, "cosine_similarity": 0.88}
        ]
        with patch("services.retriever._semantic_search", new_callable=AsyncMock) as mock_search, \
             patch("services.retriever._rerank") as mock_rerank, \
             patch("services.retriever.embed_query", new_callable=AsyncMock) as mock_embed:
            mock_search.return_value = mock_candidates
            mock_embed.return_value = [0.1] * 1536
            mock_rerank.return_value = [
                RetrievedChunk("相關內容", "test.pdf", 0, 0.88, 9.0, "相關內容")
            ]
            from services.retriever import retrieve
            results = await retrieve("查詢", "tenant_a")

        mock_rerank.assert_called_once()
        assert len(results) == 1
