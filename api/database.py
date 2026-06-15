from __future__ import annotations

import asyncpg
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

# 向量維度需與 embedding 供應商一致：
#   OpenAI text-embedding-3-small → 1536
#   Google text-embedding-004     → 768
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# 全域連線池
_pool: asyncpg.Pool | None = None


async def init_db(database_url: str) -> None:
    """建立連線池並初始化資料表。已初始化則直接返回（冪等，供 CLI 腳本重複呼叫）。"""
    global _pool
    if _pool is not None:
        return
    _pool = await asyncpg.create_pool(
        dsn=database_url,
        min_size=2,
        max_size=10,
        command_timeout=60,
    )
    await _create_tables()
    logger.info("Database pool initialized")


async def close_db() -> None:
    """關閉連線池。"""
    global _pool
    if _pool:
        await _pool.close()
        logger.info("Database pool closed")


async def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db() first.")
    return _pool


@asynccontextmanager
async def get_conn() -> AsyncGenerator[asyncpg.Connection, None]:
    """取得單一連線的 context manager，供 router 使用。"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


async def _create_tables() -> None:
    """建立 pgvector extension 與所需資料表。"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        # 啟用 pgvector
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 文件 chunks 表（含向量欄位）
        # 向量維度由 EMBEDDING_DIM 決定（OpenAI=1536, Google=768）
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id          SERIAL PRIMARY KEY,
                document_id UUID        NOT NULL,
                tenant_id   VARCHAR(100) NOT NULL,
                filename    TEXT        NOT NULL,
                chunk_index INT         NOT NULL,
                content     TEXT        NOT NULL,
                embedding   vector({EMBEDDING_DIM}),
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # 向量索引（HNSW，適合 cosine 相似度）
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON document_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)

        # tenant + document 組合索引，加速過濾
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_tenant
            ON document_chunks (tenant_id, document_id);
        """)

        logger.info("Tables and indexes created (or already exist)")
