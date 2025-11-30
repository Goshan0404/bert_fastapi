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

RUN mkdir bert
RUN mkdir tokenizer

RUN curl "https://drive.usercontent.google.com/download?id=1tWmikxMBnfLr61nHevav6Bi6NETsOMTO&confirm=xxx" -o ./bert/model.safetensors
RUN curl "https://drive.usercontent.google.com/download?id=1oHUrb_DPtij0CAdjPQNGupStpuSNpxJt&confirm=xxx" -o ./bert/config.json


RUN curl "https://drive.usercontent.google.com/download?id=1VfdVpqio4vN2Pvc7Aey7y88g5b8oVLgZ&confirm=xxx" -o ./tokenizer/special_tokens_map.json
RUN curl "https://drive.usercontent.google.com/download?id=1d_0HZ1h4P7H55LIv5lPbNU6WQDREfKIt&confirm=xxx" -o ./tokenizer/tokenizer.json
RUN curl "https://drive.usercontent.google.com/download?id=1Hn1N3aQronIHU4mC369dk_FW2qUMZwrE&confirm=xxx" -o ./tokenizer/tokenizer_config.json
RUN curl "https://drive.usercontent.google.com/download?id=1pOxTulpaq8fTJbWM6VtNiDvawGkU7mkD&confirm=xxx" -o ./tokenizer/vocab.txt



EXPOSE 5050

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050"]
