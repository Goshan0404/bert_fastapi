from typing import List

from fastapi import FastAPI
from fastapi import FastAPI, UploadFile

import torch
import pandas as pd
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("./tokenizer",  repo_type="local_files")
model = AutoModelForSequenceClassification.from_pretrained("./bert")


class Review(BaseModel):
    text: str

class Label(BaseModel):
    label: str
    confidence: float


all_labels = ['Positive', 'Negative', 'Zalupa']
app = FastAPI()

@app.post("/labels/file")
def get_labels(file: UploadFile):
    if not file:
        return {"message": "No upload file sent"}

    try:
        df = pd.read_csv(file.file)


        # проверяем, что есть колонка text
        if "text" not in df.columns:
            return {"exception": "CSV must contain a 'text' column"}

        # получаем список строк
        texts = df["text"].astype(str).tolist()

        # токенизация
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = model(**inputs)
            max_indices = torch.argmax(outputs.logits, dim=1)
            max_probs = torch.softmax(outputs.logits, dim=1).max(dim=1).values
            res = [Label(label=all_labels[i], confidence=p.item()) for i, p in zip(max_indices, max_probs)]

            return {"labels": res}

    except Exception as e:
        return {"exception": f"failed to read: {e}"}



@app.post("/labels/string")
def get_labels(reviews: List[Review]):
    if len(reviews) == 0:
        return {"exception": "empty reviews"}

    reviews_data = jsonable_encoder(reviews)

    texts = [r['text'] for r in reviews_data]
    inputs = tokenizer(texts,  padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        max_indices = torch.argmax(outputs.logits, dim=1)
        max_probs = torch.softmax(outputs.logits, dim=1).max(dim=1).values
        res = [Label(label=all_labels[i], confidence=p.item()) for i, p in zip(max_indices, max_probs)]

        return {"labels": res}





