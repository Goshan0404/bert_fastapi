from pydantic import BaseModel


class Review(BaseModel):
    text: str
