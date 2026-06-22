# DocuMind — 企業內部文件智能問答系統

> Python · FastAPI · RAG + Reranker · LangGraph Agent · Docker · pgvector

企業文件不能丟進 ChatGPT，但員工又需要快速查詢 SOP、法規、技術文件。  
DocuMind 讓你安全地把文件留在自己的資料庫，用 RAG + Agent 回答問題。

🔗 **線上 Demo**：https://documind-production-7465.up.railway.app  
（部署於 Railway，內含聊天前端；選部門 → 直接提問。例：「員工到職滿兩年有幾天特別休假？」）

---

## 架構

```
內建聊天前端 (單頁 Web UI，由 FastAPI 直接 serve)
        │ HTTP / SSE
        ▼
FastAPI (Python 3.11)
        │
        ├─ POST /api/v1/documents   上傳 → 解析 → Embedding → pgvector
        ├─ POST /api/v1/chat        問答 → LangGraph Agent → SSE streaming
        └─ POST /api/v1/eval        RAGAS 評估
        │
LangGraph Agent
        ├─ classifier_node   判斷問題是否需要追問（模糊則主動釐清）
        ├─ retriever_node    pgvector 語意搜尋 + 多語言 CrossEncoder Reranker
        └─ generator_node    Gemini / Claude / GPT 生成 + 來源引用
        │
PostgreSQL + pgvector        向量儲存，tenant_id 多租戶隔離
                             （+ _shared 全公司共用命名空間，各部門皆可查）
```

---

## 快速啟動

```bash
# 1. Clone 專案
git clone https://github.com/yaoqizha/documind
cd documind

# 2. 設定 API key
cp .env.example .env        # 填入 GOOGLE_API_KEY（見下方環境變數）

# 3. 一鍵啟動（Docker Compose：postgres + pgvector + api）
docker compose up -d --build

# 4. 開啟聊天介面（或 /docs 看 API）
open http://localhost:8000
```

---

## 環境變數

複製 `.env.example` 為 `.env`，填入你要用的供應商 API key（預設用 Google Gemini，免費額度友善，於 https://aistudio.google.com/apikey 申請）：

| 變數 | 說明 | 預設值 |
|------|------|--------|
| `GOOGLE_API_KEY` | Google Gemini API key | — |
| `ANTHROPIC_API_KEY` | Claude API key（可選） | — |
| `OPENAI_API_KEY` | OpenAI API key（可選） | — |
| `LLM_PROVIDER` | `google` / `anthropic` / `openai` | `google` |
| `LLM_MODEL` | 生成模型 | `gemini-2.5-flash-lite` |
| `EMBEDDING_PROVIDER` | `google` 或 `openai` | `google` |
| `EMBEDDING_MODEL` | Embedding 模型 | `models/gemini-embedding-001` |
| `EMBEDDING_DIM` | 向量維度（gemini-embedding-001 降維至 768；OpenAI=1536） | `768` |
| `DATABASE_URL` | PostgreSQL 連線字串 | docker-compose 預設 |
| `RETRIEVER_TOP_K` | 語意搜尋候選數 | `10` |
| `RERANKER_TOP_N` | Reranker 最終保留數 | `3` |
| `RERANKER_MODEL` | Reranker 模型（多語言，中文友善） | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| `SHARED_TENANT` | 全公司共用文件的租戶名 | `_shared` |

> ⚠️ 切換 embedding 供應商會改變向量維度，需重建 `document_chunks` 表（刪除 volume 或 drop table 後重啟）。

---

## API 使用範例

### 上傳文件
```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F 'file=@company_policy.pdf' \
  -F 'tenant_id=hr_dept'
```

### 問答（同步）
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"query": "員工年假幾天？", "tenant_id": "hr_dept"}'
```

### 問答（SSE streaming）
```javascript
const res = await fetch('/api/v1/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: '員工年假幾天？', tenant_id: 'hr_dept' })
})
const reader = res.body.getReader()
// 接收 {"type":"answer_chunk","content":"..."} 事件
```

---

## RAGAS 評估

### 跑評估
```bash
# 在 api 目錄下
python -m eval.run_evaluation my_tenant 20
```

### 指標與實測（量測 → 發現問題 → 修正 → 再量測）

同一組 5 題測試集、相同生成/評審模型（`gemini-2.5-flash-lite`），**只更換 Reranker**前後對照：

| 指標 | 目標 | Before（英文 reranker） | After（多語 reranker） |
|------|------|------|------|
| `faithfulness` | > 0.85 | 0.634 | **0.722** ↑ |
| `answer_relevancy` | > 0.80 | 0.687 | **0.867** ✅ |
| `context_recall` | > 0.75 | 0.700 | **1.000** ✅ |

> 量測條件：5 題自動生成測試集、含 CrossEncoder Reranker（2026-06）。報告產出於 `api/eval/reports/`。

### 關鍵發現與工程迭代

- **量測發現「檢索盲點」**：baseline 中，中文問題的相關段落常未被選進 top-3（例：問「特別休假」卻檢索到「加班/遠距」），導致回答「找不到」，三項指標被明顯拉低。
- **定位根因**：原 Reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` 為**英文訓練**，對中文相關性評分失準。
- **修正**：改用多語言、輕量的 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`（可由 `RERANKER_MODEL` 設定）。
- **再量測驗證**：`context_recall` 0.700 → **1.000**、`answer_relevancy` 0.687 → **0.867**（達標），`faithfulness` 0.634 → 0.722。**證實 Reranker 選型是中文 RAG 品質的關鍵槓桿。**
- **後續方向**：以更強模型（`gemini-2.5-flash`/`pro`）當 judge、擴大測試集（20+ 題）、強化生成 prompt 的忠實度約束，進一步推升 faithfulness。

---

## 測試

```bash
cd api

# document parser（無需外部依賴）
pytest tests/test_document_parser.py -v

# retriever（mock CrossEncoder）
pytest tests/test_retriever.py -v

# agent nodes（mock LLM）
pytest tests/test_agent.py -v

# 全部
pytest tests/ -v
```

---

## 部署到雲端（Railway）

本專案已部署於 Railway（[線上 Demo](https://documind-production-7465.up.railway.app)）。重點步驟與實戰經驗：

1. **資料庫**：用 `pgvector` 模板部署 Postgres（內建 pgvector extension，啟動時自動建立向量索引）。
2. **API 服務**：從 GitHub repo 部署，**Root Directory 設為 `api`**（Dockerfile 在此）。
3. **環境變數**（API 服務）：
   ```
   GOOGLE_API_KEY=<你的金鑰>
   LLM_PROVIDER=google
   LLM_MODEL=gemini-2.5-flash-lite
   EMBEDDING_PROVIDER=google
   EMBEDDING_MODEL=models/gemini-embedding-001
   EMBEDDING_DIM=768
   RERANK_ENABLED=false
   # DATABASE_URL 引用 pgvector 的原始變數（避免巢狀引用無法跨服務解析）：
   DATABASE_URL=postgresql://${{pgvector.POSTGRES_USER}}:${{pgvector.POSTGRES_PASSWORD}}@${{pgvector.RAILWAY_PRIVATE_DOMAIN}}:5432/${{pgvector.POSTGRES_DB}}
   ```

### 部署實戰筆記（踩過的坑）
- **動態埠**：容器須監聽 `$PORT`（Dockerfile 用 `--port ${PORT:-8000}`），產生 Domain 時 target port 要與實際監聽埠一致。
- **私有網路啟動時序**：容器剛啟動時 `*.railway.internal` 可能尚未就緒，故 `init_db` 內建**連線重試**。
- **`DATABASE_URL` 引用**：直接引用 `${{pgvector.DATABASE_URL}}` 會因其值本身含巢狀引用而解析為空；改引用 pgvector 的原始變數（user/password/private-domain/db）即可。
- **低資源 reranker**：免費層 CPU/RAM 跑 torch CrossEncoder 會 OOM／極慢，故線上設 `RERANK_ENABLED=false`，改用 embedding 餘弦相似度取 top-N（本機保留完整 reranker 與 RAGAS 評估）。

---

## 專案結構

```
documind/
├── api/
│   ├── main.py                  FastAPI 進入點
│   ├── database.py              asyncpg 連線池 + pgvector 初始化
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── models/
│   │   └── schemas.py           Pydantic v2 型別定義
│   ├── routers/
│   │   ├── documents.py         上傳 / 列表 / 刪除文件
│   │   ├── chat.py              問答（同步 + SSE streaming）
│   │   └── eval.py              觸發 RAGAS 評估
│   ├── services/
│   │   ├── document_parser.py   PDF/MD/TXT 解析 + 分塊
│   │   ├── embeddings.py        Google / OpenAI Embedding（批次 + 重試）
│   │   ├── retriever.py         pgvector 語意搜尋 + 多語言 CrossEncoder Reranker
│   │   ├── agent.py             LangGraph 三節點 Agent
│   │   └── prompts.py           所有 prompt template
│   ├── static/
│   │   └── index.html           單頁聊天前端（SSE 串流、來源引用、部門切換）
│   ├── eval/
│   │   ├── generate_testset.py  自動生成測試問答對
│   │   └── run_evaluation.py    RAGAS 評估執行器
│   └── tests/
│       ├── test_document_parser.py   14 個單元測試 ✅
│       ├── test_retriever.py         9 個單元測試（含 mock）✅
│       └── test_agent.py             9 個單元測試（含 mock）✅  （全部共 32 個通過）
├── docker-compose.yml
├── .env.example
├── TESTING.md                   手動測試清單
├── sample_docs/                 範例文件（HR + 全公司共用）
├── test_files/                  測試用文件（IT 部門）
└── scripts/
    └── setup.sh                 一鍵環境設定
```
