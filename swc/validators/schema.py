from typing import List, Union

from pydantic import BaseModel


class Validation(BaseModel):
    title: str
    status: bool
    errors: List[str] = []
