ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    sox \
    git \
    build-essential \
    ca-certificates \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app


ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    HF_HOME=/root/.cache/huggingface \
    PATH="/app/.venv/bin:$PATH"

ARG WORKER_GROUP

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-cache-dir --no-dev --no-install-project --group ${WORKER_GROUP} && uv cache clean


COPY src ./src
RUN uv sync --frozen --no-cache-dir --no-dev --group ${WORKER_GROUP}


CMD ["echo", "hello, perhaps you did no specified command using docker-compose?"]