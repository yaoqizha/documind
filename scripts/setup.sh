#!/usr/bin/env bash
# DocuMind 一鍵環境設定腳本
# 使用方式：bash scripts/setup.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[setup]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $1"; }
err()  { echo -e "${RED}[error]${NC} $1"; exit 1; }

# ── 檢查必要工具 ─────────────────────────────────────────────
log "Checking prerequisites..."
command -v docker   >/dev/null 2>&1 || err "Docker not found. Install: https://docs.docker.com/get-docker/"
command -v python3  >/dev/null 2>&1 || err "Python 3 not found."
python3 -c "import sys; assert sys.version_info >= (3,11)" 2>/dev/null || \
    warn "Python 3.11+ recommended (found $(python3 --version))"

# ── .env 設定 ────────────────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    log "Created .env from .env.example"
    echo ""
    warn "Please edit .env and add your API keys before continuing:"
    warn "  OPENAI_API_KEY=sk-..."
    warn "  or"
    warn "  ANTHROPIC_API_KEY=sk-ant-..."
    echo ""
    read -p "Press Enter after editing .env to continue..." _
fi

# ── 啟動 Docker 服務 ─────────────────────────────────────────
log "Starting Docker services (postgres + api)..."
docker compose up -d --build

# 等待 postgres 就緒
log "Waiting for PostgreSQL to be ready..."
for i in $(seq 1 30); do
    if docker compose exec postgres pg_isready -U documind -d documind >/dev/null 2>&1; then
        log "PostgreSQL is ready!"
        break
    fi
    if [ "$i" -eq 30 ]; then
        err "PostgreSQL did not start in 30 seconds. Check: docker compose logs postgres"
    fi
    sleep 1
done

# ── 安裝 Python 依賴（本地開發用）─────────────────────────────
log "Installing Python dependencies..."
cd api
python3 -m pip install -r requirements.txt --quiet
cd ..

# ── 跑基礎測試 ───────────────────────────────────────────────
log "Running unit tests..."
cd api
python3 -m pytest tests/test_document_parser.py -v --tb=short
cd ..

# ── 完成 ─────────────────────────────────────────────────────
echo ""
log "Setup complete!"
echo ""
echo "  API:      http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  Health:   http://localhost:8000/health"
echo ""
echo "  Quick test:"
echo "    curl http://localhost:8000/health"
echo ""
echo "  Upload a document:"
echo "    curl -X POST http://localhost:8000/api/v1/documents \\"
echo "      -F 'file=@yourfile.pdf' \\"
echo "      -F 'tenant_id=my_company'"
echo ""
echo "  Ask a question:"
echo "    curl -X POST http://localhost:8000/api/v1/chat \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"query\": \"請說明請假規定\", \"tenant_id\": \"my_company\"}'"
