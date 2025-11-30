FROM python:3.10-slim AS base

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

COPY --from=base /usr/local /usr/local

COPY . .

EXPOSE 5050

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050"]
