"""
RAGAS 評估執行器。
指標：faithfulness, answer_relevancy, context_recall
目標：faithfulness > 0.85, answer_relevancy > 0.80, context_recall > 0.75
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
REPORTS_DIR = Path("eval/reports")


async def evaluate_tenant(
    tenant_id: str,
    num_questions: int = 20,
) -> tuple[dict, str]:
    """
    完整評估流程：
    1. 生成測試集（或讀取已有的）
    2. 對每個問題呼叫 RAG pipeline 取得回答
    3. 用 RAGAS 計算指標
    4. 儲存報告並回傳

    Returns: (metrics_dict, report_path)
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 以 CLI 方式執行時不會經過 FastAPI lifespan，需自行初始化 DB 連線池
    from database import init_db
    await init_db(os.getenv(
        "DATABASE_URL",
        "postgresql://documind:documind_pass@localhost:5432/documind",
    ))

    # Step 1: 取得測試集
    testset_path = Path(f"eval/testset_{tenant_id}.json")
    if testset_path.exists():
        logger.info(f"Loading existing testset from {testset_path}")
        testset = json.loads(testset_path.read_text())
    else:
        logger.info(f"Generating new testset for tenant={tenant_id}")
        from eval.generate_testset import generate_testset
        testset = await generate_testset(tenant_id, num_questions)
        testset_path.write_text(json.dumps(testset, ensure_ascii=False, indent=2))

    if not testset:
        raise ValueError("Empty testset. Cannot evaluate.")

    # Step 2: 呼叫 RAG 取得回答（批次）
    logger.info(f"Running RAG on {len(testset)} questions...")
    from services.agent import get_agent
    agent = get_agent()

    questions, answers, contexts, ground_truths = [], [], [], []
    for item in testset:
        state = await agent.ainvoke({
            "query": item["question"],
            "tenant_id": tenant_id,
            "needs_clarification": False,
            "clarification_question": "",
            "retrieved_chunks": [],
            "answer": "",
            "node_trace": [],
        })
        questions.append(item["question"])
        answers.append(state.get("answer", ""))
        contexts.append([c.content for c in state.get("retrieved_chunks", [])]
                        or item.get("contexts", []))
        ground_truths.append(item["ground_truth"])

    # Step 3: RAGAS 評估
    logger.info("Computing RAGAS metrics...")
    metrics = await _compute_ragas(questions, answers, contexts, ground_truths)

    # Step 4: 儲存報告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"eval_{tenant_id}_{timestamp}.json"
    report = {
        "tenant_id": tenant_id,
        "evaluated_at": datetime.utcnow().isoformat(),
        "num_questions": len(questions),
        "metrics": metrics,
        "passed": (
            metrics["faithfulness"] > 0.85
            and metrics["answer_relevancy"] > 0.80
            and metrics["context_recall"] > 0.75
        ),
        "details": [
            {
                "question": q,
                "answer": a,
                "ground_truth": gt,
                "contexts_count": len(ctx),
            }
            for q, a, gt, ctx in zip(questions, answers, ground_truths, contexts)
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info(f"Report saved to {report_path}")
    logger.info(f"Metrics: {metrics}")

    return metrics, str(report_path)


async def _compute_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """使用 RAGAS 計算三個核心指標。"""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.metrics import faithfulness, answer_relevancy, context_recall

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(data)

        # RAGAS 預設用 OpenAI；依設定改用對應供應商的 LLM 與 embeddings
        ragas_llm, ragas_emb = _build_ragas_models()
        # 免費層每分鐘僅 10 次請求，完全序列化並加大重試與退避，避免爆量 429
        run_config = RunConfig(max_workers=1, timeout=300, max_retries=12, max_wait=60)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_recall],
            llm=ragas_llm,
            embeddings=ragas_emb,
            run_config=run_config,
        )
        # RAGAS 0.2.x 的 result[metric] 可能回傳 per-sample 分數列表，需自行聚合平均
        return {
            "faithfulness": _aggregate(result, "faithfulness"),
            "answer_relevancy": _aggregate(result, "answer_relevancy"),
            "context_recall": _aggregate(result, "context_recall"),
        }
    except Exception as e:
        logger.error(f"RAGAS computation failed: {e}", exc_info=True)
        # 回傳 placeholder，讓系統不崩潰
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "_error": str(e),
        }


def _aggregate(result, key: str) -> float:
    """從 RAGAS 結果取出指標分數並平均（容忍 scalar / list / NaN）。"""
    import math
    try:
        val = result[key]
    except Exception:
        return 0.0
    if isinstance(val, (list, tuple)):
        nums = [x for x in val if isinstance(x, (int, float)) and not math.isnan(x)]
        return round(sum(nums) / len(nums), 4) if nums else 0.0
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return 0.0


def _build_ragas_models():
    """依環境變數建立 RAGAS 用的 LLM 與 embeddings（支援 openai / anthropic / google）。"""
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    llm_provider = os.getenv("LLM_PROVIDER", "anthropic")
    emb_provider = os.getenv("EMBEDDING_PROVIDER", "openai")

    if llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        lc_llm = ChatAnthropic(
            model=os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=2048,
        )
    elif llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        lc_llm = ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_output_tokens=4096,  # 2.5 系列含思考 token，加大避免 LLMDidNotFinish
        )
    else:
        from langchain_openai import ChatOpenAI
        lc_llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    if emb_provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        lc_emb = GoogleGenerativeAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        lc_emb = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    return _make_ragas_llm(lc_llm, llm_provider), LangchainEmbeddingsWrapper(lc_emb)


def _make_ragas_llm(lc_llm, provider: str):
    """包裝 RAGAS LLM。Google 供應商需特別處理 temperature 不相容問題。"""
    from ragas.llms import LangchainLLMWrapper

    if provider != "google":
        return LangchainLLMWrapper(lc_llm)

    class _GeminiSafeLLM(LangchainLLMWrapper):
        # langchain-google-genai 2.0.9 會把 runtime temperature 直接丟給 google client
        # 導致 "unexpected keyword argument 'temperature'"，故覆寫為不轉傳該參數。
        def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            return self.langchain_llm.generate_prompt([prompt], stop=stop, callbacks=callbacks)

        async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            return await self.langchain_llm.agenerate_prompt(
                [prompt], stop=stop, callbacks=callbacks
            )

    return _GeminiSafeLLM(lc_llm)


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    tenant = sys.argv[1] if len(sys.argv) > 1 else "default"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    metrics, path = asyncio.run(evaluate_tenant(tenant, n))
    print(f"\n{'='*50}")
    print(f"RAGAS Evaluation Results — tenant: {tenant}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        target = {"faithfulness": 0.85, "answer_relevancy": 0.80, "context_recall": 0.75}.get(k)
        status = "PASS" if target and v > target else "FAIL" if target else ""
        print(f"  {k:25s}: {v:.4f}  {status}")
    print(f"\nReport: {path}")
