import re
from typing import List

import pandas as pd
from fastapi.responses import FileResponse
import torch
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from pandas import DataFrame
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from models.Label import Label
from models.Review import Review

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = './bert'
tokenizer_path = './tokenizer'
app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,  repo_type="local_files")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

index_to_label = {0: 'Нейтральный', 1: 'Положительный', 2: 'Отрицательный'}


def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)        # удаляем html
    text = re.sub(r"\s+", " ", text).strip()   # нормализуем пробелы
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"[\x00-\x1F\x7F]+", " ", text)
    text = re.sub(r"([^\w\s])\1{2,}", r"\1\1", text)
    text = re.sub(r"[^\w\s.,!?:;\"'()\-…/]+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"\s*n\s*", " ", text)

    return text

def preprocess(texts, head=100, tail=100, max_length=300):
    processed = []

    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]

        if len(ids) > head + tail + 10:  # запас
            head_tokens = ids[:head]
            tail_tokens = ids[-tail:]
            mid_token = tokenizer.sep_token_id
            new_ids = head_tokens + [mid_token] + tail_tokens
        else:
            new_ids = ids

        encoded = tokenizer.prepare_for_model(
            new_ids,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )

        processed.append(encoded)

    # Склеиваем дикты в батч
    batch = {
        "input_ids": torch.tensor([p["input_ids"] for p in processed]),
        "attention_mask": torch.tensor([p["attention_mask"] for p in processed])
    }

    return batch



@app.get('/')
def root():
    return "Get out of here"

@app.post("/labels")
def get_labels(review: str):
    # if len(reviews) == 0:
    #     raise HTTPException(status_code=404, detail="Empty list")
    # reviews_data = jsonable_encoder(reviews)

    texts = [clean_text(review)]

    inputs = preprocess(texts)

    with torch.no_grad():
        outputs = model(**inputs)
        max_indices = torch.argmax(outputs.logits, dim=1)
        max_probs = torch.softmax(outputs.logits, dim=1).max(dim=1).values
        df = DataFrame()
        df['labels'] = pd.Series(max_indices.cpu().numpy()).map(index_to_label)
        df['text'] = review
        df['confidence'] = max_probs
        path = 'predictions.csv'
        df.to_csv(path)

        return FileResponse(path=path)


@app.post("/labels/file")
def get_labels(file: UploadFile):
    if not file:
        raise HTTPException(status_code=404, detail="File is not uploaded")

    try:
        import csv
        df = pd.read_csv(
            file.file,
            sep=",")

        if "text" not in df.columns:
            return {"exception": "Csv file lacks of text column"}

        # Чистим тексты
        df["text"] = df["text"].astype(str).apply(clean_text)

        # Токенизация пачкой
        inputs = preprocess(df['text'].tolist())

        with torch.no_grad():
            outputs = model(**inputs)
            max_indices = torch.argmax(outputs.logits, dim=1)
            max_probs = torch.softmax(outputs.logits, dim=1).max(dim=1).values

        df['labels'] = [index_to_label[int(i)] for i in max_indices]
        df['confidence'] = max_probs.cpu().numpy()

        path = 'predictions.csv'
        df.to_csv(path, index=False)

        return FileResponse(path=path)

    except Exception as e:
        return {"exception": f"failed to read: {e}"}






