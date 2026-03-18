"""
Ticket Parsing API Endpoints

Endpoints for OCR ticket parsing operations.
"""

from fastapi import APIRouter, UploadFile, File, status
from typing import Annotated

from ...dependencies import TicketServiceDep, RequestIdDep
from ...schemas.ticket import TicketParseResponse

router = APIRouter(prefix="/ticket", tags=["Ticket OCR"])


@router.post(
    "/parse",
    response_model=TicketParseResponse,
    status_code=status.HTTP_200_OK,
    summary="Parse ticket with OCR",
    description="""
    Parse a grouping order (编组单) ticket using OCR.

    Supported file formats:
    - Images: JPEG, PNG, BMP, GIF
    - Documents: PDF

    The OCR system will extract:
    - Train information (车号, 车种)
    - Route information (发站, 到站)
    - Container information (集装箱1, 集装箱2)
    - Load information (自重, 载重, 换长)
    - Cargo details (品名, 记事)
    - Document info (票据号, 计划序号)
    """,
    responses={
        200: {"description": "Ticket parsed successfully"},
        400: {"description": "Invalid file format"},
        422: {"description": "OCR processing failed"},
        502: {"description": "Upstream service error"},
    },
)
async def parse_ticket(
    service: TicketServiceDep,
    request_id: RequestIdDep,
    file: Annotated[UploadFile, File(description="Ticket image or document")],
) -> TicketParseResponse:
    """Parse ticket document using OCR and extract structured data."""
    response = await service.parse_ticket(file)
    response.request_id = request_id
    return response
