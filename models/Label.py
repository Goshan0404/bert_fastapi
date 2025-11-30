from pydantic import BaseModel


class Label(BaseModel):
    label: str
    confidence: float