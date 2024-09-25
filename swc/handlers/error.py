from typing import List

from pydantic import BaseModel


class ValidateError(BaseModel):
    title: str = ""
    status: bool = True
    errors: List[str]
