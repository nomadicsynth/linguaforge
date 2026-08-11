FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/linguaforge-env
ENV UV_PROJECT_ENVIRONMENT=/opt/linguaforge-env/.venv
ENV PATH="/opt/linguaforge-env/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project