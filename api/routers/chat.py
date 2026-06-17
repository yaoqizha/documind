import json
import logging
import math
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from models.schemas import ChatRequest, ChatResponse, Source
from services.agent import get_agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE streaming 問答。
    前端用 EventSource 或 fetch + ReadableStream 接收。

    事件格式：
      data: {"type": "node_start", "node": "classifier"}
      data: {"type": "answer_chunk", "content": "..."}
      data: {"type": "sources", "sources": [...]}
      data: {"type": "done"}
      data: {"type": "clarification", "question": "..."}
    """
    session_id = request.session_id or str(uuid.uuid4())
    agent = get_agent()

    async def event_stream():
        initial_state = {
            "query": request.query,
            "tenant_id": request.tenant_id,
            "needs_clarification": False,
            "clarification_question": "",
            "retrieved_chunks": [],
            "answer": "",
            "node_trace": [],
        }

        try:
            async for event in agent.astream_events(initial_state, version="v2"):
                event_name = event.get("event", "")
                node_name = event.get("name", "")

                # 節點開始
                if event_name == "on_chain_start" and node_name in ("classifier", "retriever", "generator"):
                    yield _sse({"type": "node_start", "node": node_name})

                # 節點完成 — 取得輸出
                if event_name == "on_chain_end":
                    output = event.get("data", {}).get("output", {})

                    if node_name == "classifier" and output.get("needs_clarification"):
                        yield _sse({
                            "type": "clarification",
                            "question": output.get("clarification_question", ""),
                            "session_id": session_id,
                        })
                        return  # 追問後結束，等使用者回應

                    if node_name == "generator" and output.get("answer"):
                        # 把 answer 分段串流（模擬逐字輸出）
                        answer = output["answer"]
                        for chunk in _split_into_chunks(answer, size=50):
                            yield _sse({"type": "answer_chunk", "content": chunk})

                        # 來源資訊
                        sources = [
                            {
                                "filename": c.filename,
                                "chunk_index": c.chunk_index,
                                "relevance_score": _relevance(c.rerank_score),
                                "excerpt": c.excerpt,
                            }
                            for c in output.get("retrieved_chunks", [])
                        ]
                        yield _sse({"type": "sources", "sources": sources})

            yield _sse({"type": "done", "session_id": session_id})

        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 關閉 Nginx buffering
        },
    )


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    同步問答（非 streaming）。方便測試和 RAGAS 評估使用。
    """
    session_id = request.session_id or str(uuid.uuid4())
    agent = get_agent()

    initial_state = {
        "query": request.query,
        "tenant_id": request.tenant_id,
        "needs_clarification": False,
        "clarification_question": "",
        "retrieved_chunks": [],
        "answer": "",
        "node_trace": [],
    }

    final_state = await agent.ainvoke(initial_state)

    sources = [
        Source(
            filename=c.filename,
            chunk_index=c.chunk_index,
            relevance_score=_relevance(c.rerank_score),
            excerpt=c.excerpt,
        )
        for c in final_state.get("retrieved_chunks", [])
    ]

    return ChatResponse(
        answer=final_state.get("answer", ""),
        sources=sources,
        session_id=session_id,
        clarification_asked=final_state.get("needs_clarification", False),
        clarification_question=final_state.get("clarification_question") or None,
    )


# ── Helpers ───────────────────────────────────────────────────

def _relevance(logit: float) -> float:
    """CrossEncoder logit → 0–1 相關度（sigmoid），供前端顯示更直覺。"""
    return round(1.0 / (1.0 + math.exp(-logit)), 4)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _split_into_chunks(text: str, size: int = 50) -> list[str]:
    """把長文字切成固定大小的片段，模擬 token streaming。"""
    return [text[i:i + size] for i in range(0, len(text), size)]
