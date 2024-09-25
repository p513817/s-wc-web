from typing import Optional

from . import schema


def verify_mock_ssd(mock_ssd: Optional[str] = "") -> schema.Validation:
    errors = []
    if mock_ssd == "":
        errors.append("Mock SSD name is null")
    return schema.Validation(title="ssd", status=len(errors) != 0, errors=errors)
