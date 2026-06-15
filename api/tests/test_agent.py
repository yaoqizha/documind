"""
單元測試：services/agent.py
執行：pytest tests/test_agent.py -v
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.agent import (
    classifier_node,
    retriever_node,
    generator_node,
    route_after_classifier,
    AgentState,
)
from services.retriever import RetrievedChunk


def _make_state(**kwargs) -> AgentState:
    """建立測試用的初始 AgentState。"""
    defaults = {
        "query": "測試問題",
        "tenant_id": "test_tenant",
        "needs_clarification": False,
        "clarification_question": "",
        "retrieved_chunks": [],
        "answer": "",
        "node_trace": [],
    }
    defaults.update(kwargs)
    return AgentState(**defaults)


# ── classifier_node 測試 ─────────────────────────────────────

@pytest.mark.asyncio
class TestClassifierNode:
    async def test_clear_question_no_clarification_needed(self):
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "needs_clarification": False,
            "clarification_question": "",
            "reason": "問題清晰",
        })
        with patch("services.agent._get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_fn.return_value = mock_llm

            state = _make_state(query="公司的請假規定是什麼？")
            result = await classifier_node(state)

        assert result["needs_clarification"] is False
        assert result["clarification_question"] == ""
        assert "classifier" in result["node_trace"]

    async def test_ambiguous_question_needs_clarification(self):
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "needs_clarification": True,
            "clarification_question": "請問您詢問的是哪個部門的最新規定？",
            "reason": "問題包含模糊時間詞",
        })
        with patch("services.agent._get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_fn.return_value = mock_llm

            state = _make_state(query="最新的規定是什麼？")
            result = await classifier_node(state)

        assert result["needs_clarification"] is True
        assert "clarification_question" in result
        assert result["clarification_question"] != ""

    async def test_invalid_llm_response_defaults_to_no_clarification(self):
        mock_response = MagicMock()
        mock_response.content = "這不是合法的 JSON 格式"
        with patch("services.agent._get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_fn.return_value = mock_llm

            state = _make_state()
            result = await classifier_node(state)

        assert result["needs_clarification"] is False  # 預設不追問


# ── retriever_node 測試 ──────────────────────────────────────

@pytest.mark.asyncio
class TestRetrieverNode:
    async def test_retriever_node_populates_chunks(self):
        mock_chunks = [
            RetrievedChunk("相關內容", "doc.pdf", 0, 0.9, 8.5, "相關內容")
        ]
        with patch("services.agent.retrieve", new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = mock_chunks
            state = _make_state()
            result = await retriever_node(state)

        assert result["retrieved_chunks"] == mock_chunks
        assert "retriever" in result["node_trace"]

    async def test_retriever_node_with_no_results(self):
        with patch("services.agent.retrieve", new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = []
            state = _make_state()
            result = await retriever_node(state)

        assert result["retrieved_chunks"] == []


# ── generator_node 測試 ──────────────────────────────────────

@pytest.mark.asyncio
class TestGeneratorNode:
    async def test_generator_with_chunks(self):
        mock_chunks = [
            RetrievedChunk("公司年假為 14 天", "hr_policy.pdf", 0, 0.9, 8.5, "公司年假為 14 天")
        ]
        mock_response = MagicMock()
        mock_response.content = "根據 HR 政策文件，公司年假為 14 天。【來源：hr_policy.pdf】"

        with patch("services.agent._get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_fn.return_value = mock_llm

            state = _make_state(retrieved_chunks=mock_chunks)
            result = await generator_node(state)

        assert "14 天" in result["answer"]
        assert "generator" in result["node_trace"]

    async def test_generator_without_chunks_returns_fallback(self):
        state = _make_state(retrieved_chunks=[])
        result = await generator_node(state)

        assert result["answer"] != ""
        assert "找不到" in result["answer"] or "抱歉" in result["answer"]


# ── route_after_classifier 測試 ──────────────────────────────

class TestRouting:
    def test_routes_to_retriever_when_no_clarification(self):
        state = _make_state(needs_clarification=False)
        assert route_after_classifier(state) == "retriever"

    def test_routes_to_end_when_clarification_needed(self):
        state = _make_state(needs_clarification=True)
        assert route_after_classifier(state) == "end_clarify"
