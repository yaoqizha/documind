from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    content: str
    chunk_index: int
    source_page: int | None = None   # PDF 頁碼（若有）


def parse_document(filename: str, file_bytes: bytes) -> list[TextChunk]:
    """
    依副檔名選擇解析器，回傳切好的 TextChunk 列表。
    支援：.pdf, .md, .txt
    """
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        raw_pages = _parse_pdf(file_bytes)
    elif ext in (".md", ".txt"):
        raw_pages = [(file_bytes.decode("utf-8", errors="replace"), None)]
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .md, .txt")

    chunks: list[TextChunk] = []
    for text, page_num in raw_pages:
        page_chunks = _split_text(text)
        for chunk_text in page_chunks:
            if chunk_text.strip():
                chunks.append(TextChunk(
                    content=chunk_text.strip(),
                    chunk_index=len(chunks),
                    source_page=page_num,
                ))

    logger.info(f"Parsed '{filename}' → {len(chunks)} chunks")
    return chunks


def _parse_pdf(file_bytes: bytes) -> list[tuple[str, int]]:
    """回傳 [(page_text, page_number), ...]。"""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf not installed. Run: pip install pypdf")

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, i + 1))
    return pages


def _split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    RecursiveCharacterTextSplitter 的純 Python 實作。
    優先在段落（\n\n）、換行（\n）、句號、空格切割。
    """
    separators = ["\n\n", "\n", "。", ".", " ", ""]
    return _recursive_split(text, separators, chunk_size, chunk_overlap)


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = separators[0] if separators else ""
    remaining_seps = separators[1:] if separators else []

    if sep and sep in text:
        parts = text.split(sep)
    else:
        if remaining_seps:
            return _recursive_split(text, remaining_seps, chunk_size, chunk_overlap)
        # 最後手段：強制切
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = (current + sep + part).lstrip(sep) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)
            # 如果 part 本身超過 chunk_size，遞迴切
            if len(part) > chunk_size:
                sub = _recursive_split(part, remaining_seps, chunk_size, chunk_overlap)
                chunks.extend(sub)
                # overlap：把最後一塊的尾巴帶進 current
                current = sub[-1][-chunk_overlap:] if sub else ""
            else:
                current = part

    if current.strip():
        chunks.append(current)

    return chunks
