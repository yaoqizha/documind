import logging
import os
from dataclasses import dataclass

from database import get_conn
from services.embeddings import embed_query

logger = logging.getLogger(__name__)

TOP_K = int(os.getenv("RETRIEVER_TOP_K", "10"))
TOP_N = int(os.getenv("RERANKER_TOP_N", "3"))
# 全公司共用文件的租戶；任何部門查詢時都會一併檢索到此命名空間
SHARED_TENANT = os.getenv("SHARED_TENANT", "_shared")
# 多語言 CrossEncoder：mmarco-mMiniLMv2 對中文重排顯著優於英文版 ms-marco-MiniLM，
# 且體積輕量（~470MB），對部署環境的記憶體友善。
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
# Reranker 開關：低資源部署環境（如 Railway 免費層）可設 RERANK_ENABLED=false，
# 改用 embedding 餘弦相似度直接取 top_n，避免載入 torch 導致 OOM／推論過慢。
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() not in ("false", "0", "no")

# CrossEncoder 模型（首次呼叫時懶載入）
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading CrossEncoder reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker model loaded")
    return _reranker


@dataclass
class RetrievedChunk:
    content: str
    filename: str
    chunk_index: int
    semantic_score: float    # pgvector cosine 距離（越小越近）
    rerank_score: float      # CrossEncoder 分數（越高越相關）
    excerpt: str             # 前 200 字，供前端顯示


async def retrieve(
    query: str,
    tenant_id: str,
    top_k: int = TOP_K,
    top_n: int = TOP_N,
) -> list[RetrievedChunk]:
    """
    兩階段檢索：
    1. pgvector cosine 語意搜尋，取 top_k 候選
    2. CrossEncoder Reranker 重排序，取 top_n 最終結果
       （RERANK_ENABLED=false 時跳過第 2 步，直接取餘弦相似度前 top_n）
    """
    # Step 1: 語意檢索
    query_vec = await embed_query(query)
    candidates = await _semantic_search(query_vec, tenant_id, top_k)

    if not candidates:
        logger.warning(f"No candidates found for tenant={tenant_id}, query='{query[:50]}'")
        return []

    # Step 2: Reranker（可關閉）
    if RERANK_ENABLED:
        reranked = _rerank(query, candidates, top_n)
        logger.info(f"Retrieved {len(candidates)} candidates → reranked to {len(reranked)}")
    else:
        # 不重排：candidates 已按 cosine 距離排序，直接取前 top_n
        reranked = [
            RetrievedChunk(
                content=c["content"],
                filename=c["filename"],
                chunk_index=c["chunk_index"],
                semantic_score=float(c["cosine_similarity"]),
                rerank_score=float(c["cosine_similarity"]),
                excerpt=c["content"][:200],
            )
            for c in candidates[:top_n]
        ]
        logger.info(f"Retrieved {len(candidates)} candidates → top {len(reranked)} (rerank disabled)")
    return reranked


async def _semantic_search(
    query_vec: list[float],
    tenant_id: str,
    top_k: int,
) -> list[dict]:
    """pgvector cosine 距離搜尋（<=> 運算子）。"""
    # pgvector 格式：'[0.1, 0.2, ...]'
    vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"

    # 同時檢索該租戶與共用租戶（_shared）；共用文件全部門可見
    tenants = [tenant_id]
    if tenant_id != SHARED_TENANT:
        tenants.append(SHARED_TENANT)

    sql = """
        SELECT
            content,
            filename,
            chunk_index,
            1 - (embedding <=> $1::vector) AS cosine_similarity
        FROM document_chunks
        WHERE tenant_id = ANY($2::varchar[])
        ORDER BY embedding <=> $1::vector
        LIMIT $3
    """
    async with get_conn() as conn:
        rows = await conn.fetch(sql, vec_str, tenants, top_k)

    return [dict(r) for r in rows]


def _rerank(
    query: str,
    candidates: list[dict],
    top_n: int,
) -> list[RetrievedChunk]:
    """CrossEncoder Reranker：(query, passage) → 相關性分數。"""
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, c["content"]) for c in candidates]
    scores = reranker.predict(pairs)
    # CrossEncoder 回傳 numpy array；測試 mock 可能回傳 list，兩者皆相容
    scores = scores.tolist() if hasattr(scores, "tolist") else list(scores)

    scored = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )[:top_n]

    results = []
    for score, c in scored:
        results.append(RetrievedChunk(
            content=c["content"],
            filename=c["filename"],
            chunk_index=c["chunk_index"],
            semantic_score=float(c["cosine_similarity"]),
            rerank_score=float(score),
            excerpt=c["content"][:200],
        ))
    return results
