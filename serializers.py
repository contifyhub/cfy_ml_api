from pydantic import BaseModel


class NerText(BaseModel):
    text: list


class ArticleText(BaseModel):
    text: list

