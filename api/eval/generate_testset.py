"""
測試集生成器：從已索引文件中自動生成問答對，供 RAGAS 評估使用。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


async def generate_testset(tenant_id: str, num_questions: int = 20) -> list[dict]:
    """
    從 pgvector 中抽取 chunk，請 LLM 生成問答對。
    回傳格式：[{"question": ..., "ground_truth": ..., "context": [...]}]
    """
    from database import get_conn, init_db
    await init_db(os.getenv("DATABASE_URL", "postgresql://documind:documind_pass@localhost:5432/documind"))

    # 隨機抽取 chunks（每個問題用 2-3 個 chunk 生成）
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT content, filename
            FROM document_chunks
            WHERE tenant_id = $1
            ORDER BY RANDOM()
            LIMIT $2
            """,
            tenant_id,
            num_questions * 3,
        )

    if not rows:
        raise ValueError(f"No documents found for tenant_id='{tenant_id}'. Upload documents first.")

    chunks = [dict(r) for r in rows]
    testset = []

    # 每次用 2-3 個 chunk 生成 1 個問答對
    for i in range(num_questions):
        context_chunks = chunks[i * 2: i * 2 + 2]
        if not context_chunks:
            break
        context_text = "\n---\n".join(c["content"] for c in context_chunks)
        qa = await _generate_qa_pair(context_text)
        if qa:
            testset.append({
                "question": qa["question"],
                "ground_truth": qa["answer"],
                "contexts": [c["content"] for c in context_chunks],
                "source_files": list({c["filename"] for c in context_chunks}),
            })

    logger.info(f"Generated {len(testset)} QA pairs for tenant={tenant_id}")
    return testset


async def _generate_qa_pair(context: str) -> dict | None:
    """請 LLM 根據 context 生成一個問答對。"""
    prompt = f"""根據以下文件內容，生成一個有意義的問題與標準答案。

文件內容：
{context[:1000]}

回答格式（只輸出 JSON）：
{{"question": "具體問題", "answer": "根據文件的完整答案"}}

規則：
- 問題要具體，能從文件中找到答案
- 不要問「這份文件是關於什麼的」這類泛泛問題
- 答案要根據文件內容，不要推測
"""
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            msg = await client.messages.create(
                model=os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            content = msg.content[0].text
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
            llm = ChatGoogleGenerativeAI(
                model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                # gemini-2.5 系列有內建思考會佔用 token，需給足額度避免 JSON 被截斷
                max_output_tokens=2048,
            )
            resp = await llm.ainvoke([HumanMessage(content=prompt)])
            content = resp.content
        else:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = await client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            content = resp.choices[0].message.content

        # 清理 markdown code block
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        # 穩健解析：擷取第一個 { 到最後一個 }，避免前後雜訊或思考文字導致解析失敗
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start:end + 1]
        return json.loads(content)

    except Exception as e:
        logger.warning(f"QA generation failed: {e}")
        return None


if __name__ == "__main__":
    import sys
    tenant = sys.argv[1] if len(sys.argv) > 1 else "default"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    result = asyncio.run(generate_testset(tenant, n))
    out_path = Path(f"eval/testset_{tenant}.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved {len(result)} QA pairs to {out_path}")
