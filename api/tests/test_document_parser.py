"""
單元測試：services/document_parser.py
執行：pytest tests/test_document_parser.py -v
"""
import pytest
from services.document_parser import parse_document, _split_text, TextChunk


# ── _split_text 測試 ──────────────────────────────────────────

class TestSplitText:
    def test_short_text_returns_single_chunk(self):
        text = "這是一段很短的文字。"
        result = _split_text(text, chunk_size=500)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_splits_into_multiple_chunks(self):
        # 建立一段超過 chunk_size 的文字
        text = "這是一個段落。\n\n" * 50
        result = _split_text(text, chunk_size=100, chunk_overlap=20)
        assert len(result) > 1

    def test_chunk_size_respected(self):
        text = "A" * 1000
        chunk_size = 200
        result = _split_text(text, chunk_size=chunk_size)
        for chunk in result:
            assert len(chunk) <= chunk_size + 50  # 容許少許超出（因分隔符）

    def test_empty_text_returns_empty_list(self):
        result = _split_text("", chunk_size=500)
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        result = _split_text("   \n\n   ", chunk_size=500)
        assert result == [] or all(not c.strip() for c in result)

    def test_paragraph_split_preferred(self):
        # 有 \n\n 分段時，應優先在段落間切割
        text = "第一段內容。" * 20 + "\n\n" + "第二段內容。" * 20
        result = _split_text(text, chunk_size=200, chunk_overlap=20)
        assert any("\n\n" not in chunk for chunk in result), "應在段落邊界切割"


# ── parse_document 測試 ───────────────────────────────────────

class TestParseDocument:
    def test_parse_txt_file(self):
        content = "這是測試文字檔案的內容。\n\n第二段落的內容在這裡。"
        result = parse_document("test.txt", content.encode("utf-8"))
        assert len(result) >= 1
        assert all(isinstance(c, TextChunk) for c in result)

    def test_parse_md_file(self):
        content = "# 標題\n\n這是 Markdown 文件內容。\n\n## 小節\n\n更多內容。"
        result = parse_document("test.md", content.encode("utf-8"))
        assert len(result) >= 1

    def test_chunk_index_is_sequential(self):
        content = "段落一內容。\n\n" * 30
        result = parse_document("test.txt", content.encode("utf-8"))
        indices = [c.chunk_index for c in result]
        assert indices == list(range(len(result))), "chunk_index 必須從 0 開始連續遞增"

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_document("test.docx", b"some content")

    def test_chunks_have_content(self):
        content = "有意義的文字內容。" * 100
        result = parse_document("test.txt", content.encode("utf-8"))
        assert all(c.content.strip() for c in result), "所有 chunk 必須有非空內容"

    def test_large_file_splits_correctly(self):
        # 模擬大型文件
        paragraphs = [f"第 {i} 段落的詳細內容，包含一些具體說明。" for i in range(200)]
        content = "\n\n".join(paragraphs)
        result = parse_document("large.txt", content.encode("utf-8"))
        assert len(result) >= 10, "大文件應該有多個 chunks"
        assert len(result) < 500, "不應該有過多細碎 chunks"


# ── TextChunk dataclass 測試 ─────────────────────────────────

class TestTextChunk:
    def test_text_chunk_creation(self):
        chunk = TextChunk(content="測試內容", chunk_index=0, source_page=1)
        assert chunk.content == "測試內容"
        assert chunk.chunk_index == 0
        assert chunk.source_page == 1

    def test_text_chunk_default_page_is_none(self):
        chunk = TextChunk(content="測試", chunk_index=0)
        assert chunk.source_page is None
