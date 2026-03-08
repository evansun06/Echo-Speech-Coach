#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
  set +a
fi

if [ -f "$ROOT_DIR/.env.gemini" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env.gemini"
  set +a
fi

export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-config.settings.dev}"
export DB_HOST="${HYBRID_DB_HOST:-localhost}"
export DB_PORT="${HYBRID_DB_PORT:-5433}"
export CELERY_BROKER_URL="${HYBRID_CELERY_BROKER_URL:-redis://localhost:6379/0}"
export CELERY_RESULT_BACKEND="${HYBRID_CELERY_RESULT_BACKEND:-redis://localhost:6379/1}"
export LLM_LEDGER_REDIS_URL="${HYBRID_LLM_LEDGER_REDIS_URL:-$CELERY_BROKER_URL}"
export MEDIA_ROOT="${HYBRID_MEDIA_ROOT:-$BACKEND_DIR/media}"
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/gcloud/application_default_credentials.json}"
export HYBRID_CELERY_POOL="${HYBRID_CELERY_POOL:-solo}"
export HYBRID_CELERY_CONCURRENCY="${HYBRID_CELERY_CONCURRENCY:-1}"
mkdir -p "$MEDIA_ROOT"

if [ -n "${HYBRID_PYTHON_BIN:-}" ]; then
  PYTHON_BIN="$HYBRID_PYTHON_BIN"
elif [ -x "$ROOT_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [ -x "$BACKEND_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$BACKEND_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

cd "$BACKEND_DIR"
exec "$PYTHON_BIN" -m celery -A config worker \
  -l "${CELERY_WORKER_LOGLEVEL:-info}" \
  -E \
  --pool "${HYBRID_CELERY_POOL}" \
  --concurrency "${HYBRID_CELERY_CONCURRENCY}" \
  "$@"
