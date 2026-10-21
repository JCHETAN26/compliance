# syntax=docker/dockerfile:1.7
# One image recipe for every Python service; SERVICE_PACKAGE selects which workspace
# member (and its dependencies) is installed into the virtualenv.
ARG PYTHON_IMAGE=python:3.11-slim-bookworm

FROM ghcr.io/astral-sh/uv:0.8.22 AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
ARG SERVICE_PACKAGE
RUN test -n "${SERVICE_PACKAGE}" || (echo "SERVICE_PACKAGE build arg is required" >&2; exit 1)
COPY pyproject.toml uv.lock .python-version ./
COPY packages ./packages
COPY apps/api ./apps/api
COPY apps/worker ./apps/worker
COPY apps/model-service ./apps/model-service
COPY apps/mcp-server ./apps/mcp-server
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package "${SERVICE_PACKAGE}"

FROM ${PYTHON_IMAGE} AS runtime
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /home/app --create-home app \
    && mkdir -p /data \
    && chown app:app /data
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data
USER app
WORKDIR /home/app
