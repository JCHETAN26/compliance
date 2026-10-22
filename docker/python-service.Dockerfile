# syntax=docker/dockerfile:1.7
# One recipe for every Python service; SERVICE_PACKAGE selects the workspace member and
# DEPS_PACKAGE the (shareable) third-party dependency layer.
#
# Disk-conscious layout (the worker carries CPU-only PyTorch, ~1 GB):
# - single runtime stage: the virtualenv is built in place, never copied between stages
#   (a builder->runtime COPY would store the venv twice);
# - uv and the sources are bind-mounted, not copied, so they are not image layers;
# - third-party dependencies are installed from a manifests-only stage (lockfile +
#   pyproject files), so source edits do not reinstall them;
# - no uv cache is kept (UV_NO_CACHE) and bytecode is compiled lazily at runtime.
ARG PYTHON_IMAGE=python:3.11-slim-bookworm

FROM ghcr.io/astral-sh/uv:0.8.22 AS uv

FROM scratch AS manifests
COPY pyproject.toml uv.lock /src/
COPY packages/core/pyproject.toml /src/packages/core/
COPY packages/llm/pyproject.toml /src/packages/llm/
COPY packages/ingestion/pyproject.toml /src/packages/ingestion/
COPY packages/retrieval/pyproject.toml /src/packages/retrieval/
COPY packages/agents/pyproject.toml /src/packages/agents/
COPY apps/api/pyproject.toml /src/apps/api/
COPY apps/worker/pyproject.toml /src/apps/worker/
COPY apps/model-service/pyproject.toml /src/apps/model-service/
COPY apps/mcp-server/pyproject.toml /src/apps/mcp-server/
COPY seed/pyproject.toml /src/seed/

FROM ${PYTHON_IMAGE} AS runtime
# Space-separated Debian packages (the worker needs tesseract-ocr for OCR).
ARG SYSTEM_PACKAGES=""
RUN if [ -n "${SYSTEM_PACKAGES}" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends ${SYSTEM_PACKAGES} \
      && rm -rf /var/lib/apt/lists/*; \
    fi
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /home/app --create-home app \
    && mkdir -p /data/storage /cache \
    && chown -R app:app /data /cache
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=0
# Third-party dependency layer. DEPS_PACKAGE selects whose dependency set is installed;
# services built with the same DEPS_PACKAGE and SYSTEM_PACKAGES share every layer up to
# and including this one (the worker and model-service share the ~1.3 GB CPU-PyTorch
# layer, so it is stored once). SERVICE_PACKAGE is declared only *after* this layer:
# BuildKit passes every declared ARG to later RUN steps, so declaring it earlier would
# make otherwise identical layers differ per service.
ARG DEPS_PACKAGE
RUN test -n "${DEPS_PACKAGE}" || (echo "DEPS_PACKAGE build arg is required" >&2; exit 1)
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=bind,from=manifests,source=/src,target=/src,rw \
    cd /src && uv sync --frozen --no-dev --no-install-workspace --package "${DEPS_PACKAGE}"
ARG SERVICE_PACKAGE
RUN test -n "${SERVICE_PACKAGE}" || (echo "SERVICE_PACKAGE build arg is required" >&2; exit 1)
# --inexact keeps the shared dependency layer intact (no removals for services that
# need a subset of it); only this service's missing packages and the workspace
# members are added.
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=bind,source=.,target=/src,rw \
    cd /src && uv sync --frozen --no-dev --no-editable --inexact --package "${SERVICE_PACKAGE}"
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data \
    STORAGE_DIR=/data/storage \
    ARTIFACT_CACHE_DIR=/cache
USER app
WORKDIR /home/app
