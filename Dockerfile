# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# системные зависимости (нужны для torch/catboost/xgboost в зависимости от окружения)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Poetry
RUN pip install --no-cache-dir "poetry>=2.0.0,<3.0.0"

# Сначала только метаданные для кеша слоёв
COPY pyproject.toml poetry.lock* /app/

# Установка зависимостей (без dev)
RUN poetry install --no-root

# Теперь код проекта (включая пакет common)
COPY . /app/

# Установка самого проекта (чтобы common гарантированно был установлен как package)
RUN poetry install --only main

# Если это FastAPI сервис, можно оставить так (переопредели в docker-compose при необходимости)
# ЗАМЕНИ common.api:app на реальный модуль:переменная приложения
EXPOSE 8000
CMD ["uvicorn", "common.api:app", "--host", "0.0.0.0", "--port", "8002"]
