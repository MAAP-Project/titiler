"""
This code was pulled from stac_pydantic:
https://github.com/stac-utils/stac-pydantic/blob/master/stac_pydantic/utils.py
"""

from enum import Enum
from typing import Any, List


class AutoValueEnum(Enum):
    """
    An auto-incrementing enumerated value
    """

    def _generate_next_value_(  # type: ignore
        name: str, start: int, count: int, last_values: List[Any]
    ) -> Any:
        return name
