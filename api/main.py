import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Settings ─────────────────────────────────────────────────

class Settings(BaseSettings):
    database_url: str = "postgresql://documind:documind_pass@localhost:5432/documind"
    app_env: str = "development"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


# ── Lifespan（startup / shutdown）────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting DocuMind API [{settings.app_env}]")
    from database import init_db, close_db
    await init_db(settings.database_url)
    logger.info("Database ready")
    yield
    await close_db()
    logger.info("DocuMind API shutdown")


# ── App ───────────────────────────────────────────────────────

app = FastAPI(
    title="DocuMind API",
    description="企業內部文件智能問答系統 — RAG + LangGraph Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────

from routers.documents import router as documents_router
from routers.chat import router as chat_router
from routers.eval import router as eval_router

app.include_router(documents_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(eval_router, prefix="/api/v1")


# ── Health check ──────────────────────────────────────────────

@app.get("/health")
async def health():
    from database import get_pool
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {e}"
    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "database": db_status,
        "version": "1.0.0",
    }


# ── Frontend（單頁聊天 UI，由 FastAPI 直接 serve）──────────────

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    """回傳聊天前端頁面。"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
