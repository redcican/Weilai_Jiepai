"""
Container Identification Schemas

Schemas for container and carriage identification operations.
"""

from pydantic import Field, field_validator
from .base import BaseSchema, ResponseSchema


class ContainerCreateRequest(BaseSchema):
    """
    Request to report container identification.

    Used when OCR identifies container numbers from images.
    """

    train_no: str = Field(
        ...,
        min_length=1,
        max_length=50,
        alias="trainNo",
        description="车号 (Train number)",
        examples=["T001", "K1234"],
    )
    carbin_no: str = Field(
        ...,
        min_length=1,
        max_length=50,
        alias="carbinNo",
        description="车厢编号 (Carriage number)",
        examples=["C001", "W12345"],
    )
    container_no: str = Field(
        ...,
        min_length=1,
        max_length=50,
        alias="contrainerNo",  # Note: matches original API typo
        description="集装箱编号 (Container number)",
        examples=["CONT001", "CSLU1234567"],
    )

    @field_validator("train_no", "carbin_no", "container_no")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field cannot be empty")
        return v


class ContainerData(BaseSchema):
    """Data returned after container identification."""

    id: int | None = Field(default=None, description="Record ID")
    train_no: str = Field(..., alias="trainNo")
    carbin_no: str = Field(..., alias="carbinNo")
    container_no: str = Field(..., alias="contrainerNo")


class ContainerResponse(ResponseSchema[ContainerData | None]):
    """Response for container identification."""

    pass


# Batch operations
class ContainerBatchItem(BaseSchema):
    """Single item for batch container reporting."""

    train_no: str = Field(..., alias="trainNo")
    carbin_no: str = Field(..., alias="carbinNo")
    container_no: str = Field(..., alias="contrainerNo")

    @field_validator("train_no", "carbin_no", "container_no")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field cannot be empty")
        return v


class ContainerBatchRequest(BaseSchema):
    """Request for batch container reporting."""

    items: list[ContainerBatchItem] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of containers to report",
    )
