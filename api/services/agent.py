import json
import logging
import os
from typing import TypedDict, AsyncIterator

from langgraph.graph import StateGraph, END

from services.retriever import retrieve, RetrievedChunk
from services.prompts import (
    CLASSIFIER_SYSTEM, CLASSIFIER_USER,
    GENERATOR_SYSTEM, GENERATOR_USER,
    NO_CONTEXT_RESPONSE,
)

logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")


# ── State Definition ─────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    tenant_id: str
    # Classifier output
    needs_clarification: bool
    clarification_question: str
    # Retriever output
    retrieved_chunks: list[RetrievedChunk]
    # Generator output
    answer: str
    # Debug
    node_trace: list[str]


# ── LLM Factory ──────────────────────────────────────────────

def _get_llm():
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=LLM_MODEL,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=2048,
        )
    elif LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_output_tokens=2048,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=2048,
        )


# ── Helpers ───────────────────────────────────────────────────

def _parse_json_loose(text: str):
    """
    從 LLM 回應穩健解析 JSON：擷取第一個 { 到最後一個 }，
    容忍 markdown code fence（```json ... ```）與前後雜訊。
    解析失敗回傳 None。
    """
    if not isinstance(text, str):
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


# ── Node Functions ────────────────────────────────────────────

async def classifier_node(state: AgentState) -> AgentState:
    """
    判斷問題是否模糊，決定要不要先追問。
    輸出：needs_clarification, clarification_question
    """
    logger.debug(f"[classifier_node] query='{state['query'][:60]}'")
    llm = _get_llm()

    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=CLASSIFIER_SYSTEM),
        HumanMessage(content=CLASSIFIER_USER.format(query=state["query"])),
    ]
    response = await llm.ainvoke(messages)

    result = _parse_json_loose(response.content)
    if result is not None:
        state["needs_clarification"] = bool(result.get("needs_clarification", False))
        state["clarification_question"] = result.get("clarification_question", "")
    else:
        # 解析失敗就繼續往下走，不中斷流程
        state["needs_clarification"] = False
        state["clarification_question"] = ""

    state["node_trace"].append("classifier")
    return state


async def retriever_node(state: AgentState) -> AgentState:
    """
    pgvector 語意搜尋 + CrossEncoder Reranker。
    輸出：retrieved_chunks
    """
    logger.debug(f"[retriever_node] tenant={state['tenant_id']}")
    chunks = await retrieve(
        query=state["query"],
        tenant_id=state["tenant_id"],
    )
    state["retrieved_chunks"] = chunks
    state["node_trace"].append("retriever")
    return state


async def generator_node(state: AgentState) -> AgentState:
    """
    根據 retrieved_chunks 生成回答。
    輸出：answer
    """
    chunks = state["retrieved_chunks"]

    if not chunks:
        state["answer"] = NO_CONTEXT_RESPONSE
        state["node_trace"].append("generator(no_context)")
        return state

    # 組裝 context，每個 chunk 標注來源
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[片段 {i}] 來源：{chunk.filename}\n{chunk.content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    llm = _get_llm()
    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=GENERATOR_SYSTEM),
        HumanMessage(content=GENERATOR_USER.format(
            context=context,
            query=state["query"],
        )),
    ]
    response = await llm.ainvoke(messages)
    state["answer"] = response.content
    state["node_trace"].append("generator")
    return state


# ── Routing ───────────────────────────────────────────────────

def route_after_classifier(state: AgentState) -> str:
    """如果需要追問就結束（由 API 層處理），否則繼續檢索。"""
    if state["needs_clarification"]:
        return "end_clarify"
    return "retriever"


# ── Graph Assembly ────────────────────────────────────────────

def build_agent() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("classifier", classifier_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)

    graph.set_entry_point("classifier")
    graph.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {
            "end_clarify": END,
            "retriever": "retriever",
        },
    )
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", END)

    return graph.compile()


# Singleton agent（避免每次請求重新組裝 graph）
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent
