FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ src/
RUN uv sync --frozen --no-dev

COPY config/ config/
COPY scripts/ scripts/
COPY Makefile .

CMD ["uv", "run", "python", "scripts/train_model.py", "--synthetic"]
