# One image for the API, the dashboard and the monitoring commands.
# Data, models and the SQLite database are mounted at runtime (docker-compose.yml).
FROM python:3.12.10-slim-bookworm

# PYTHONPATH: import the code straight from /app (no pip install of the project),
# so monitoring/config.py resolves the project root to /app
# (database -> /app/db, model -> /app/models).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Non-root user that owns only the database folder.
RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --create-home --home-dir /home/app app

# Dependencies first, so code changes do not reinstall them.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml ./
COPY src/ src/
COPY api/ api/
COPY dashboard/ dashboard/
COPY docker/ docker/
COPY tests/ tests/

# Mount points. /app/db must belong to `app` before a new named volume is first
# mounted there, so Docker gives the volume the same owner.
RUN mkdir -p db models data/processed \
    && chown app:app db

USER app

EXPOSE 8000 8502

# Default: the API. docker-compose.yml sets the command for each service.
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
