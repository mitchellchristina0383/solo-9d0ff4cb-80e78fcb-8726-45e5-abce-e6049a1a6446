# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_DATA_DIR=/data \
    APP_UPLOAD_DIR=/data/uploads \
    APP_DB_PATH=/data/trainer.sqlite3 \
    APP_MAX_UPLOAD_MB=512 \
    PORT=8000

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY trainer ./trainer
COPY wsgi.py ./
COPY tests ./tests
COPY scripts ./scripts

RUN useradd --create-home --uid 10001 trainer \
    && mkdir -p /data/uploads \
    && chown -R trainer:trainer /data /srv
USER trainer

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/' % os.environ.get('PORT','8000'), timeout=4)"

# APP_SECRET_KEY is required for session cookies (flash messages); compose passes
# it in.  A container started without it falls back to Flask's dev default.
# One worker process: training runs in a background thread and the pause/status
# bookkeeping uses process-local thread handles.  Extra gunicorn workers would
# still share the SQLite checkpoints, but the "is a training thread running"
# answer would be wrong in the other processes.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 8 --timeout 120 --graceful-timeout 10 --no-control-socket --access-logfile - wsgi:application"]
