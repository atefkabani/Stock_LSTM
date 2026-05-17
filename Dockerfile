# syntax=docker/dockerfile:1.6

# ---------- Base image ----------
# Slim Python image keeps the layer small; TensorFlow CPU wheels work on 3.11.
FROM python:3.11-slim AS base

# ---------- Environment ----------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ---------- System packages ----------
# curl is used by the healthcheck; build-essential is only needed for some
# transitive deps and is removed after install to keep the image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------- App user ----------
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

# ---------- Python deps (cached layer) ----------
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ---------- Application code ----------
# Copy the rest of the project. Make sure .dockerignore excludes the
# training dataset and any local virtualenvs so the build context stays small.
COPY . .

# Ensure the app user owns the working dir
RUN chown -R app:app /app
USER app

# ---------- Networking ----------
EXPOSE 8501

# ---------- Healthcheck ----------
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ---------- Entrypoint ----------
# Default = Streamlit web app. Override with `docker run ... python predict.py ...`
# to use the CLI predictor instead.
CMD ["streamlit", "run", "app/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0"]