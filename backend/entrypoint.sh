#!/bin/sh
set -e

cd /app

echo "Applying database migrations..."
python manage.py migrate --noinput

exec "$@"
