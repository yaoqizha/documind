# DocuMind 手動測試清單

逐項打勾即可完整驗證系統。測試檔案放在 [`test_files/`](test_files/)（IT 部門用）與 [`sample_docs/`](sample_docs/)（HR 部門用）。

> 前置：服務需已啟動 → `docker compose up -d`，前端在 http://localhost:8000

---

## 0. 服務健康
- [v] `docker compose ps` → `postgres` 與 `api` 兩容器皆 `Up`
- [v] 瀏覽器開 http://localhost:8000 → 看到 DocuMind 聊天介面
- [v] 右上角狀態燈顯示綠色「● 服務正常」
- [v] `curl http://localhost:8000/health` → `{"status":"ok","database":"ok",...}`

## 1. 文件上傳（解析 → embedding → pgvector）
測試三種格式與一個新租戶（順便驗證多租戶）。

- [v] 左側「租戶」輸入框改成 `it_dept`
- [v] 上傳 `test_files/it_dept_VPN與遠端連線規範.md`（.md 格式）→ 跳出「已索引 N 個片段」
- [v] 上傳 `test_files/it_dept_密碼與帳號安全政策.txt`（.txt 格式）→ 成功
- [v] 左側「已索引文件」清單出現這兩份，顯示 chunk 數
- [ ]（選配）將任一 .txt 用 Mac「列印 → 存成 PDF」後上傳，驗證 .pdf 解析
- [ ] 嘗試上傳不支援格式（如 .docx 改名）→ 應回錯誤提示（400）

## 2. 問答 + SSE 逐字串流 + 來源引用（it_dept）
- [v] 租戶維持 `it_dept`，問：**VPN 帳號多久會開通？**
  - [v] 節點徽章依序亮起：分類 → 檢索 → 生成
  - [v] 回答**逐字串流**出現，內容含「兩個工作天」
  - [v] 回答結尾有【來源：…】引用
  - [v] 下方來源卡片顯示檔名、**相關度（0–1）**、摘錄
- [v] 問：**公司密碼最少要幾碼？** → 回答含「十二碼」
- [v] 問：**MFA 裝置遺失怎麼辦？** → 回答含「通報 IT 部門辦理重設」

## 3. 多租戶物理隔離（重點賣點）
- [v] 租戶維持 `it_dept`，問：**員工特別休假幾天？**（這是 HR 文件內容）
  - [v] 預期回「在現有文件中找不到相關資訊」——因為 it_dept 看不到 hr_dept 文件 ✅
- [v] 租戶改成 `hr_dept`，問同一題：**員工到職滿兩年有幾天特別休假？**
  - [v] 預期正確回「十日特別休假」，來源為「公司差勤管理辦法」 ✅
- [v] 反向再驗一次：`hr_dept` 問 **VPN 帳號多久開通？** → 應回「找不到」

## 4. Agent 主動追問（classifier node）
- [v] 租戶 `hr_dept`，問模糊問題：**最新的規定是什麼？**（含模糊詞「最新」）
  - [v] 預期出現黃色追問框，反問你要查哪方面 🤔
  - [v]（註：是否追問由模型判斷，偶爾會直接回答；可多試「那個規定怎麼算？」）

## 5. 重排品質（中文 Reranker）
- [v] `hr_dept` 問：**業務交際費每人每次上限多少？** → 回「三千元」，來源為「員工報銷與差旅費辦法」且相關度最高
- [v] `hr_dept` 問：**記過幾次會被免職？** → 回「三次」，來源為「績效考核與獎懲辦法」

## 6. API 直接測試（給工程面試官看）
- [v] 開 http://localhost:8000/docs → Swagger UI 可展開各 API
- [v] 同步問答：
  ```bash
  curl -X POST http://localhost:8000/api/v1/chat \
    -H 'Content-Type: application/json' \
    -d '{"query":"員工特別休假幾天？","tenant_id":"hr_dept"}'
  ```
- [v] 文件列表：`curl "http://localhost:8000/api/v1/documents?tenant_id=hr_dept"`

## 7. 自動化測試 + 評估
- [ ] 單元測試：`docker compose exec api python -m pytest tests/ -v` → 32 passed
- [ ] RAGAS 評估（需 Gemini 額度）：
  ```bash
  docker compose exec -e LLM_MODEL=gemini-2.5-flash-lite api \
    python -m eval.run_evaluation hr_dept 5
  ```
  → 產出報告於 `api/eval/reports/`，context_recall、answer_relevancy 應達標

---

### 疑難排解
- 第一次問答很慢（30–90 秒）= 容器在下載 Reranker 模型，屬正常，之後變快
- 出現 `spending cap` / `quota` 429 = Gemini 額度/支出上限問題，到 https://ai.studio/spend 調整
- 改了 `.env` 要讓 API 生效 → `docker compose up -d --force-recreate api`
