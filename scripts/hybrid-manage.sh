#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"

if [ "$#" -eq 0 ]; then
  echo "Usage: scripts/hybrid-manage.sh <manage.py args...>" >&2
  exit 1
fi

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
  set +a
fi

export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-config.settings.dev}"
export DB_HOST="${HYBRID_DB_HOST:-localhost}"
export DB_PORT="${HYBRID_DB_PORT:-5433}"
export CELERY_BROKER_URL="${HYBRID_CELERY_BROKER_URL:-redis://localhost:6379/0}"
export CELERY_RESULT_BACKEND="${HYBRID_CELERY_RESULT_BACKEND:-redis://localhost:6379/1}"
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/gcloud/application_default_credentials.json}"

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
exec "$PYTHON_BIN" manage.py "$@"
