# One image for every service (tracking server, bootstrap job, model server, UI).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet \
    HF_HOME=/cache/hf

WORKDIR /app

# Dependencies first, so code changes don't reinstall torch.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ui --no-install-project

COPY src ./src
COPY app ./app
COPY assets ./assets
COPY .streamlit ./.streamlit
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ui

RUN useradd --create-home app && mkdir -p /cache/hf /mlflow && chown -R app /cache /mlflow
USER app
