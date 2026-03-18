"""
Ticket Parsing Schemas

Schemas for OCR ticket parsing operations.
"""

from datetime import datetime
from pydantic import Field
from .base import BaseSchema, ResponseSchema


class TicketParseRequest(BaseSchema):
    """
    Request to parse a ticket using OCR.

    The ticket file is provided separately via multipart form data.
    This schema is for any additional parsing options.
    """

    # Optional parsing hints
    language: str | None = Field(
        default="zh",
        description="OCR language hint",
        examples=["zh", "en"],
    )


class TicketData(BaseSchema):
    """
    Parsed ticket data from OCR recognition.

    Contains all fields extracted from the grouping order (编组单).
    """

    # Core identifiers
    assign_id: int | None = Field(default=None, alias="ASSIGN_ID", description="记录ID")
    plan_no: str | None = Field(default=None, alias="planNo", description="计划序号")

    # Train information
    train_no: str | None = Field(default=None, alias="trainNo", description="车号")
    train_type: str | None = Field(default=None, alias="trainType", description="车种")

    # Carriage information
    stock: str | None = Field(default=None, description="股道")
    seq: str | None = Field(default=None, description="序号")

    # Load information
    oil_type: str | None = Field(default=None, alias="oilType", description="油种")
    empty_capacity: str | None = Field(default=None, alias="emptyCapacity", description="自重")
    change_length: str | None = Field(default=None, alias="changeLength", description="换长")
    load_capacity: str | None = Field(default=None, alias="loadCapacity", description="载重")

    # Container information
    container1: str | None = Field(default=None, description="集装箱1")
    container2: str | None = Field(default=None, description="集装箱2")

    # Route information
    start_station: str | None = Field(default=None, alias="startStation", description="发站")
    dest_station: str | None = Field(default=None, alias="destStation", description="到站")

    # Cargo information
    carry_type: str | None = Field(default=None, alias="carryType", description="品名")
    descr: str | None = Field(default=None, description="记事")

    # Document information
    ticket_no: str | None = Field(default=None, alias="ticketNo", description="票据号")
    attribution: str | None = Field(default=None, description="属性")
    receive_person: str | None = Field(default=None, alias="receivePerson", description="收货人")

    # Media
    pic: str | None = Field(default=None, description="照片路径")

    # Audit fields
    create_by: str | None = Field(default=None, description="创建人")
    create_time: datetime | None = Field(default=None, description="创建时间")
    update_by: str | None = Field(default=None, description="修改人")
    update_time: datetime | None = Field(default=None, description="修改时间")
    deleted: int = Field(default=0, description="删除标识")

    @property
    def route(self) -> str:
        """Get formatted route string."""
        if self.start_station and self.dest_station:
            return f"{self.start_station} → {self.dest_station}"
        return self.start_station or self.dest_station or ""

    @property
    def containers(self) -> list[str]:
        """Get list of non-empty container numbers."""
        return [c for c in [self.container1, self.container2] if c]


class TicketParseResponse(ResponseSchema[list[TicketData]]):
    """Response for ticket parsing operation."""

    @property
    def tickets(self) -> list[TicketData]:
        """Convenience accessor for ticket data."""
        return self.data or []


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
