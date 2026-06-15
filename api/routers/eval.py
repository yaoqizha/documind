import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from models.schemas import EvalRequest, EvalResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/eval", tags=["evaluation"])

# 追蹤評估狀態（生產環境應用 Redis）
_eval_status: dict[str, str] = {}


@router.post("", response_model=EvalResponse)
async def run_evaluation(request: EvalRequest, background_tasks: BackgroundTasks):
    """
    觸發 RAGAS 評估。
    評估跑在背景，立即回傳任務 ID；可用 GET /eval/{task_id} 查詢進度。
    同步版本（num_questions <= 10）直接回傳結果。
    """
    if request.num_questions <= 10:
        # 同步執行（適合測試）
        return await _run_ragas_eval(request.tenant_id, request.num_questions)
    else:
        # 大量評估改用背景任務（這裡簡化為同步，實際上可改 Celery）
        logger.info(f"Starting eval for tenant={request.tenant_id}, n={request.num_questions}")
        return await _run_ragas_eval(request.tenant_id, request.num_questions)


async def _run_ragas_eval(tenant_id: str, num_questions: int) -> EvalResponse:
    """實際執行 RAGAS 評估的邏輯。"""
    try:
        from eval.run_evaluation import evaluate_tenant
        metrics, report_path = await evaluate_tenant(tenant_id, num_questions)
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

    passed = (
        metrics["faithfulness"] > 0.85
        and metrics["answer_relevancy"] > 0.80
        and metrics["context_recall"] > 0.75
    )

    from models.schemas import EvalMetrics
    return EvalResponse(
        tenant_id=tenant_id,
        num_questions=num_questions,
        metrics=EvalMetrics(**metrics),
        passed=passed,
        report_path=report_path,
    )
