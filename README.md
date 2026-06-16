# DocuMind — 企業內部文件智能問答系統

> Python · FastAPI · RAG + Reranker · LangGraph Agent · Docker · pgvector

企業文件不能丟進 ChatGPT，但員工又需要快速查詢 SOP、法規、技術文件。  
DocuMind 讓你安全地把文件留在自己的資料庫，用 RAG + Agent 回答問題。

---

## 架構

```
前端 (Next.js / 任意前端)
        │ HTTP / SSE
        ▼
FastAPI (Python 3.11)
        │
        ├─ POST /api/v1/documents   上傳 → 解析 → Embedding → pgvector
        ├─ POST /api/v1/chat        問答 → LangGraph Agent → SSE streaming
        └─ POST /api/v1/eval        RAGAS 評估
        │
LangGraph Agent
        ├─ classifier_node   判斷問題是否需要追問
        ├─ retriever_node    pgvector 語意搜尋 + CrossEncoder Reranker
        └─ generator_node    Gemini / Claude / GPT 生成 + 來源引用
        │
PostgreSQL + pgvector        向量儲存，tenant_id 多租戶隔離
```

---

## 快速啟動

```bash
# 1. Clone 專案
git clone https://github.com/yourname/documind
cd documind

# 2. 一鍵設定（需要 Docker + Python 3.11）
bash scripts/setup.sh

# 3. 開啟 API 文件
open http://localhost:8000/docs
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
| `LLM_MODEL` | 生成模型 | `gemini-2.5-flash` |
| `EMBEDDING_PROVIDER` | `google` 或 `openai` | `google` |
| `EMBEDDING_MODEL` | Embedding 模型 | `models/gemini-embedding-001` |
| `EMBEDDING_DIM` | 向量維度（gemini-embedding-001 降維至 768；OpenAI=1536） | `768` |
| `DATABASE_URL` | PostgreSQL 連線字串 | docker-compose 預設 |
| `RETRIEVER_TOP_K` | 語意搜尋候選數 | `10` |
| `RERANKER_TOP_N` | Reranker 最終保留數 | `3` |

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

### 指標與目標

| 指標 | 目標 | 實測（baseline） | 說明 |
|------|------|------|------|
| `faithfulness` | > 0.85 | **0.634** | 回答是否有文件根據（防幻覺） |
| `answer_relevancy` | > 0.80 | **0.687** | 回答是否切題 |
| `context_recall` | > 0.75 | **0.700** | 重要資訊是否被找到 |

> 量測條件：5 題自動生成測試集、含 CrossEncoder Reranker、生成與評審皆用 `gemini-2.5-flash-lite`（2026-06）。
> 報告產出於 `api/eval/reports/`。

### 結果分析與改進方向（誠實基準）

這是**實際量測的 baseline，尚未達標**——刻意保留真實數字，因為從中定位問題比漂亮的數字更有價值：

- **主要拉低點是「檢索盲點」**：5 題中有 1 題（外部訓練費用核准層級）所需的 chunk 未被 Reranker 選進 top-3，導致回答「找不到」，該題三項指標近乎 0，把平均明顯拉低；其餘 4 題回答正確且有依據（faithfulness 約 0.79）。
- **改進方向**：
  1. 檢索召回——擴大 `RETRIEVER_TOP_K`、調整 chunk 大小／重疊、或換更強 embedding，降低相關段落漏選機率
  2. 評審穩定度——以更強模型（`gemini-2.5-flash` / `pro`）當 RAGAS judge 並擴大測試集（20+ 題）降低單題歸零的變異
  3. 生成忠實度——強化 prompt 的「僅根據提供片段作答」約束
- **設計假設（待 A/B 量測）**：CrossEncoder Reranker 從 top-10 候選精選 top-3，預期提升 faithfulness；無 Reranker 對照組為下一步實驗。

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

```bash
# 1. 安裝 Railway CLI
npm install -g @railway/cli

# 2. 登入並部署
railway login
railway init
railway up

# 3. 設定環境變數
railway variables set GOOGLE_API_KEY=AIza...
railway variables set LLM_PROVIDER=google
railway variables set LLM_MODEL=gemini-2.5-flash
railway variables set EMBEDDING_PROVIDER=google
railway variables set EMBEDDING_MODEL=models/gemini-embedding-001
railway variables set EMBEDDING_DIM=768
railway variables set DATABASE_URL=<railway-postgres-url>
```

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
│   │   ├── retriever.py         pgvector 語意搜尋 + CrossEncoder Reranker
│   │   ├── agent.py             LangGraph 三節點 Agent
│   │   └── prompts.py           所有 prompt template
│   ├── eval/
│   │   ├── generate_testset.py  自動生成測試問答對
│   │   └── run_evaluation.py    RAGAS 評估執行器
│   └── tests/
│       ├── test_document_parser.py   14 個單元測試 ✅
│       ├── test_retriever.py         9 個單元測試（含 mock）✅
│       └── test_agent.py             9 個單元測試（含 mock）✅  （全部共 32 個通過）
├── docker-compose.yml
├── .env.example
└── scripts/
    └── setup.sh                 一鍵環境設定
```

---

## 面試展示重點

1. **RAG 評估（已實測）**：整合 RAGAS，實測 baseline faithfulness 0.634 / answer_relevancy 0.687 / context_recall 0.700；並從結果定位出「檢索盲點」與具體改進方向（見 RAGAS 評估章節）——展現量測 → 分析 → 迭代的工程能力
2. **Reranker 設計**：pgvector top-10 候選 → CrossEncoder 精選 top-3，提升回答忠實度
3. **多租戶隔離**：`WHERE tenant_id = ?` 讓不同部門文件物理隔離（已實測：跨租戶查詢回傳 0 筆）
4. **Agent 主動追問**：classifier node 判斷問題模糊度，自動決定是否追問
5. **SSE Streaming**：前端逐字顯示，不需等待完整回答
