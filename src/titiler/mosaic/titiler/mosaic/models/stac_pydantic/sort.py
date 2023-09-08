"""
This code was pulled from stac_pydantic:
https://github.com/stac-utils/stac-pydantic/blob/master/stac_pydantic/api/extensions/sort.py
"""

from enum import auto

from pydantic import BaseModel, Field

from titiler.mosaic.models.stac_pydantic.utils import AutoValueEnum


class SortDirections(str, AutoValueEnum):
    """
    The direction of the sort (Ascending or Descending)
    """

    asc = auto()
    desc = auto()


class SortExtension(BaseModel):
    """
    https://github.com/radiantearth/stac-api-spec/tree/master/extensions/sort#sort-api-extension
    """

    field: str = Field(..., alias="field", min_length=1)
    direction: SortDirections
