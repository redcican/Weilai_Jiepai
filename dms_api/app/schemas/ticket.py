"""
Ticket Parsing Schemas

Schemas for OCR ticket parsing operations.
Raw table data output — no column-name mapping, with table_type field.
"""

from typing import Any
from datetime import datetime
from pydantic import Field, ConfigDict
from .base import BaseSchema, ResponseSchema


class TicketParseRequest(BaseSchema):
    """
    Request to parse a ticket using OCR.

    The ticket file is provided separately via multipart form data.
    """

    language: str | None = Field(
        default="zh",
        description="OCR language hint",
        examples=["zh", "en"],
    )


class OCRTableData(BaseSchema):
    """
    Raw OCR table extraction result.

    table_type:
      1 — 站存车打印 (vehicle type/number in separate columns)
      2 — 集装箱编组单 (slash vehicle, container numbers)

    table_data: raw row arrays.
      Type 1 may start with {"股道": "4"} as first element.
      Type 2 rows are [seq, vehicle/number, container1, container2?, station?].
    """

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        # Allow mixed types (dict + list) in table_data
        arbitrary_types_allowed=True,
    )

    filename: str = Field(description="文件名")
    table_type: int = Field(alias="tableType", description="表格类型 (1=站存车, 2=集装箱编组单)")
    metadata: dict[str, str] = Field(default_factory=dict, description="元数据")
    table_data: list[Any] = Field(alias="tableData", default_factory=list, description="表格数据行")


class TicketParseResponse(ResponseSchema[list[OCRTableData]]):
    """Response for batch ticket parsing operation."""
    pass


# ── Legacy schemas for DMS backend compatibility ──

class TicketData(BaseSchema):
    """
    Parsed ticket data from DMS backend.
    Kept for backward compatibility with DMS backend proxy path.
    """

    assign_id: int | None = Field(default=None, alias="ASSIGN_ID", description="记录ID")
    plan_no: str | None = Field(default=None, alias="planNo", description="计划序号")
    train_no: str | None = Field(default=None, alias="trainNo", description="车号")
    train_type: str | None = Field(default=None, alias="trainType", description="车种")
    stock: str | None = Field(default=None, description="股道")
    seq: str | None = Field(default=None, description="序号")
    oil_type: str | None = Field(default=None, alias="oilType", description="油种")
    empty_capacity: str | None = Field(default=None, alias="emptyCapacity", description="自重")
    change_length: str | None = Field(default=None, alias="changeLength", description="换长")
    load_capacity: str | None = Field(default=None, alias="loadCapacity", description="载重")
    container1: str | None = Field(default=None, description="集装箱1")
    container2: str | None = Field(default=None, description="集装箱2")
    start_station: str | None = Field(default=None, alias="startStation", description="发站")
    dest_station: str | None = Field(default=None, alias="destStation", description="到站")
    carry_type: str | None = Field(default=None, alias="carryType", description="品名")
    descr: str | None = Field(default=None, description="记事")
    ticket_no: str | None = Field(default=None, alias="ticketNo", description="票据号")
    attribution: str | None = Field(default=None, description="属性")
    receive_person: str | None = Field(default=None, alias="receivePerson", description="收货人")
    pic: str | None = Field(default=None, description="照片路径")
    create_by: str | None = Field(default=None, description="创建人")
    create_time: datetime | None = Field(default=None, description="创建时间")
    update_by: str | None = Field(default=None, description="修改人")
    update_time: datetime | None = Field(default=None, description="修改时间")
    deleted: int = Field(default=0, description="删除标识")


class TicketSummary(BaseSchema):
    """Summary of parsed ticket for list views."""

    assign_id: int | None = Field(default=None, alias="ASSIGN_ID")
    plan_no: str | None = Field(default=None, alias="planNo")
    train_no: str | None = Field(default=None, alias="trainNo")
    route: str = Field(default="", description="Formatted route")
    create_time: datetime | None = None


class TicketListResponse(ResponseSchema[list[TicketSummary]]):
    """Response for listing parsed tickets."""
    pass
