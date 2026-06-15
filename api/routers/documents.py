import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from database import get_conn
from models.schemas import DocumentUploadResponse, DocumentListItem
from services.document_parser import parse_document
from services.embeddings import embed_texts

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


@router.post("", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = Form(..., min_length=1, max_length=100),
):
    """
    上傳文件 → 解析 → embedding → 存入 pgvector。
    支援 .pdf, .md, .txt。最大 20MB。
    """
    # 驗證副檔名
    from pathlib import Path
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # 讀取檔案
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 20MB)")

    logger.info(f"Processing '{file.filename}' ({len(file_bytes)} bytes) for tenant='{tenant_id}'")

    # Step 1: 解析 + 分塊
    try:
        chunks = parse_document(file.filename or "unknown", file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not chunks:
        raise HTTPException(status_code=422, detail="Could not extract text from document")

    # Step 2: 批次 embedding
    texts = [c.content for c in chunks]
    try:
        embeddings = await embed_texts(texts)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=502, detail="Embedding service error")

    # Step 3: 存入資料庫
    document_id = uuid.uuid4()
    async with get_conn() as conn:
        # 先刪除同 tenant + 同 filename 的舊版（更新文件用）
        await conn.execute(
            "DELETE FROM document_chunks WHERE tenant_id = $1 AND filename = $2",
            tenant_id, file.filename,
        )
        # 批次 INSERT
        records = [
            (
                str(document_id),
                tenant_id,
                file.filename,
                chunk.chunk_index,
                chunk.content,
                "[" + ",".join(str(v) for v in emb) + "]",
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        await conn.executemany(
            """
            INSERT INTO document_chunks
                (document_id, tenant_id, filename, chunk_index, content, embedding)
            VALUES ($1, $2, $3, $4, $5, $6::vector)
            """,
            records,
        )

    logger.info(f"Indexed {len(chunks)} chunks for document_id={document_id}")
    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename or "unknown",
        tenant_id=tenant_id,
        chunk_count=len(chunks),
    )


@router.get("", response_model=list[DocumentListItem])
async def list_documents(tenant_id: str):
    """列出指定租戶的所有已上傳文件。"""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT
                document_id::text,
                filename,
                COUNT(*) AS chunk_count,
                MIN(created_at) AS created_at
            FROM document_chunks
            WHERE tenant_id = $1
            GROUP BY document_id, filename
            ORDER BY MIN(created_at) DESC
            """,
            tenant_id,
        )
    return [
        DocumentListItem(
            document_id=row["document_id"],
            filename=row["filename"],
            chunk_count=row["chunk_count"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


@router.delete("/{filename}")
async def delete_document(filename: str, tenant_id: str):
    """刪除指定租戶的特定文件。"""
    async with get_conn() as conn:
        result = await conn.execute(
            "DELETE FROM document_chunks WHERE tenant_id = $1 AND filename = $2",
            tenant_id, filename,
        )
    deleted_count = int(result.split()[-1])
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted_chunks": deleted_count, "filename": filename}
