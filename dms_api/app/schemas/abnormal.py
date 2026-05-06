"""
Abnormal Alert Schemas

Schemas for abnormal alerts and carriage condition reporting.
"""

from pydantic import Field, field_validator
from .base import BaseSchema, ResponseSchema
from .common import AbnormalType, AbnormalStatus


class AbnormalCreateRequest(BaseSchema):
    """
    Request to create an abnormal alert.

    Used when detecting abnormal conditions during train operation.
    """

    carbin_no: str = Field(
        ...,
        min_length=1,
        max_length=50,
        alias="carbinNo",
        description="车厢编号 (Carriage number)",
        examples=["C12345", "W001"],
    )
    descr: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="告警信息 (Alert description)",
        examples=["发现异常震动", "温度过高告警"],
    )

    @field_validator("carbin_no", "descr")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field cannot be empty")
        return v


class AbnormalData(BaseSchema):
    """Data returned after creating an abnormal alert."""

    id: int | None = Field(default=None, description="Alert ID")
    carbin_no: str = Field(..., alias="carbinNo")
    descr: str


class AbnormalResponse(ResponseSchema[AbnormalData | None]):
    """Response for abnormal alert creation."""

    pass


class BadConditionCreateRequest(BaseSchema):
    """
    Request to report carriage condition (lock status, foreign objects).

    Used for reporting specific types of carriage anomalies.
    """

    carbin_no: str = Field(
        ...,
        min_length=1,
        max_length=50,
        alias="carbinNo",
        description="车厢编号 (Carriage number)",
        examples=["C12345"],
    )
    abnormal_type: AbnormalType = Field(
        ...,
        alias="abnormalType",
        description="异常类型 (1=未落锁, 2=有异物)",
    )
    is_abnormal: AbnormalStatus = Field(
        ...,
        alias="isAbnormal",
        description="是否异常 (0=无异常, 1=有异常)",
    )
    descr: str | None = Field(
        default=None,
        max_length=500,
        description="异常描述 (Optional description)",
        examples=["锁扣松动", "发现塑料袋"],
    )

    @field_validator("carbin_no")
    @classmethod
    def validate_carbin_no(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Carriage number cannot be empty")
        return v

    @property
    def type_label(self) -> str:
        """Get human-readable type label."""
        return self.abnormal_type.label

    @property
    def status_label(self) -> str:
        """Get human-readable status label."""
        return self.is_abnormal.label


class BadConditionData(BaseSchema):
    """Data returned after reporting bad condition."""

    id: int | None = Field(default=None, description="Record ID")
    carbin_no: str = Field(..., alias="carbinNo")
    abnormal_type: AbnormalType = Field(..., alias="abnormalType")
    is_abnormal: AbnormalStatus = Field(..., alias="isAbnormal")
    descr: str | None = None


class BadConditionResponse(ResponseSchema[BadConditionData | None]):
    """Response for bad condition report."""

    pass


# Batch operations
class AbnormalBatchItem(BaseSchema):
    """Single item for batch abnormal reporting."""

    carbin_no: str = Field(..., alias="carbinNo")
    descr: str


class BadConditionBatchItem(BaseSchema):
    """Single item for batch bad condition reporting."""

    carbin_no: str = Field(..., alias="carbinNo")
    abnormal_type: AbnormalType = Field(..., alias="abnormalType")
    is_abnormal: AbnormalStatus = Field(..., alias="isAbnormal")
    descr: str | None = None
