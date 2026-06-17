from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID, uuid4


# ── Document schemas ─────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    document_id: UUID
    filename: str
    tenant_id: str
    chunk_count: int
    status: str = "indexed"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentListItem(BaseModel):
    document_id: UUID
    filename: str
    chunk_count: int
    created_at: datetime
    is_shared: bool = False   # 是否為全公司共用文件（_shared 租戶）


# ── Chat schemas ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000,
                       description="使用者問題")
    tenant_id: str = Field(..., min_length=1, max_length=100,
                           description="租戶 ID，用於隔離不同部門文件")
    session_id: Optional[str] = Field(default=None,
                                      description="對話 session ID，不傳則建立新 session")


class Source(BaseModel):
    filename: str
    chunk_index: int
    relevance_score: float
    excerpt: str = Field(..., description="相關段落摘錄（前 200 字）")


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    session_id: str
    clarification_asked: bool = False
    clarification_question: Optional[str] = None


# ── Eval schemas ─────────────────────────────────────────────

class EvalRequest(BaseModel):
    tenant_id: str
    num_questions: int = Field(default=20, ge=5, le=100)


class EvalMetrics(BaseModel):
    faithfulness: float = Field(..., description="回答是否有文件根據，目標 > 0.85")
    answer_relevancy: float = Field(..., description="回答是否切題，目標 > 0.80")
    context_recall: float = Field(..., description="重要資訊是否被找到，目標 > 0.75")


class EvalResponse(BaseModel):
    tenant_id: str
    num_questions: int
    metrics: EvalMetrics
    passed: bool
    report_path: str
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
