"""
Container Identification API Endpoints

Endpoints for container and carriage identification.
"""

from fastapi import APIRouter, UploadFile, File, Form, status
from typing import Annotated

from ...dependencies import ContainerServiceDep, RequestIdDep
from ...schemas.container import ContainerResponse

router = APIRouter(prefix="/container", tags=["Container Identification"])


@router.post(
    "",
    response_model=ContainerResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Report container identification",
    description="""
    Submit OCR-identified container information.

    This endpoint is used when the system identifies:
    - Train number (车号)
    - Carriage number (车厢编号)
    - Container number (集装箱编号)

    A photo of the identification is required.
    """,
    responses={
        201: {"description": "Container identification recorded"},
        400: {"description": "Invalid request data"},
        422: {"description": "Validation error"},
        502: {"description": "Upstream service error"},
    },
)
async def create_container(
    service: ContainerServiceDep,
    request_id: RequestIdDep,
    train_no: Annotated[str, Form(alias="trainNo", description="车号 (Train number)")],
    carbin_no: Annotated[str, Form(alias="carbinNo", description="车厢编号 (Carriage number)")],
    container_no: Annotated[str, Form(alias="contrainerNo", description="集装箱编号 (Container number)")],
    file: Annotated[UploadFile, File(description="Identification photo")],
) -> ContainerResponse:
    """Submit container identification with photo evidence."""
    from ...schemas.container import ContainerCreateRequest

    request = ContainerCreateRequest(
        train_no=train_no,
        carbin_no=carbin_no,
        container_no=container_no,
    )
    response = await service.create_container(request, file)
    response.request_id = request_id
    return response
