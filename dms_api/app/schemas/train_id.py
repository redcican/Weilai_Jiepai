"""
Train ID Recognition Schemas

Schemas for station-entry vehicle identification operations.
"""

from pydantic import Field
from .base import BaseSchema, ResponseSchema


class TrainIDData(BaseSchema):
    """
    Recognized train identification data.

    Contains vehicle type and number extracted from station-entry camera images.
    """

    type: str = Field(
        default="",
        description="空挡标记，######## 表示空挡帧",
    )
    vehicle_type: str = Field(
        default="",
        alias="vehicleType",
        description="Vehicle type code (车种), e.g. C64K, C70E",
    )
    vehicle_number: str = Field(
        default="",
        alias="vehicleNumber",
        description="Vehicle number (车号), e.g. 49 31846",
    )
    confidence: float = Field(
        default=0.0,
        description="Average OCR confidence score (0.0 - 1.0)",
    )
    container: str = Field(
        default="",
        description="Best container ID (集装箱箱号), e.g. TBJU881313",
    )
    container_confidence: float = Field(
        default=0.0,
        alias="containerConfidence",
        description="Container recognition confidence (0.0 - 1.0)",
    )


class TrainIDResponse(ResponseSchema[TrainIDData]):
    """Response for single image train ID recognition."""

    pass


class TrainIDBatchItem(BaseSchema):
    """Single item in a batch recognition response."""

    filename: str = Field(..., description="Original filename")
    type: str = Field(
        default="",
        description="空挡标记，######## 表示空挡帧",
    )
    vehicle_type: str = Field(
        default="",
        alias="vehicleType",
        description="Vehicle type code (车种)",
    )
    vehicle_number: str = Field(
        default="",
        alias="vehicleNumber",
        description="Vehicle number (车号)",
    )
    confidence: float = Field(
        default=0.0,
        description="Average OCR confidence score",
    )
    container: str = Field(
        default="",
        description="Best container ID (集装箱箱号)",
    )
    container_confidence: float = Field(
        default=0.0,
        alias="containerConfidence",
        description="Container recognition confidence",
    )


class TrainIDBatchResponse(ResponseSchema[list[TrainIDBatchItem]]):
    """Response for batch train ID recognition."""

    pass


# ---------------------------------------------------------------------------
# Flatcar single-image recognition schemas
# ---------------------------------------------------------------------------

class FlatcarData(BaseSchema):
    """Recognized data from flatcar bottom-region processing."""

    vehicle_type: str = Field(
        default="",
        alias="vehicleType",
        description="Flatcar vehicle type (车型), e.g. X70, C70E",
    )
    vehicle_number: str = Field(
        default="",
        alias="vehicleNumber",
        description="Flatcar vehicle number (车号)",
    )
    confidence: float = Field(
        default=0.0,
        description="Average OCR confidence score",
    )


class FlatcarResponse(ResponseSchema[FlatcarData]):
    """Response for flatcar single-image recognition."""

    pass


# ---------------------------------------------------------------------------
# Video recognition schemas (deprecated, kept for reference)
# ---------------------------------------------------------------------------
# class VideoTrainIDData(BaseSchema): ...
# class VideoTrainIDResponse(ResponseSchema[VideoTrainIDData]): ...
