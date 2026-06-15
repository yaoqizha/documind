import asyncio
import logging
import os
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # openai | google
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
BATCH_SIZE = 100   # OpenAI 建議每批不超過 100 筆


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    批次取得 embedding 向量。
    自動分批、重試（最多 3 次，指數退避）。
    回傳順序與輸入順序一致。
    依 EMBEDDING_PROVIDER 切換供應商（openai / google）。
    """
    if EMBEDDING_PROVIDER == "google":
        return await _embed_google(texts)
    return await _embed_openai(texts)


async def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    all_embeddings: list[list[float]] = []
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        logger.debug(f"Embedding batch {batch_idx + 1}/{len(batches)} ({len(batch)} texts)")
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        # 回傳順序保證與輸入一致
        sorted_data = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend([item.embedding for item in sorted_data])

    return all_embeddings


async def _embed_google(texts: list[str]) -> list[list[float]]:
    """
    透過 Gemini REST batchEmbedContents 取得向量。
    指定 outputDimensionality=EMBEDDING_DIM（gemini-embedding-001 為 Matryoshka 模型，
    可降維；langchain wrapper 在此版本不支援該參數，故直接呼叫 REST）。
    降維後做 L2 normalize（Google 對非 3072 維的建議）。
    """
    import math
    import httpx

    api_key = os.getenv("GOOGLE_API_KEY")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"{EMBEDDING_MODEL}:batchEmbedContents?key={api_key}"
    )

    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    all_embeddings: list[list[float]] = []
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

    async with httpx.AsyncClient(timeout=60) as client:
        for batch_idx, batch in enumerate(batches):
            logger.debug(f"Embedding batch {batch_idx + 1}/{len(batches)} ({len(batch)} texts)")
            payload = {
                "requests": [
                    {
                        "model": EMBEDDING_MODEL,
                        "content": {"parts": [{"text": t}]},
                        "outputDimensionality": EMBEDDING_DIM,
                    }
                    for t in batch
                ]
            }
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            # 回傳順序與 requests 一致
            all_embeddings.extend(_normalize(e["values"]) for e in data["embeddings"])

    return all_embeddings


async def embed_query(query: str) -> list[float]:
    """單一查詢的 embedding（供檢索用）。"""
    results = await embed_texts([query])
    return results[0]
