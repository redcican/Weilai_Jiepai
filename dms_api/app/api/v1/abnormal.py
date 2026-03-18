"""
Abnormal Alert API Endpoints

Endpoints for abnormal alerts and carriage condition reporting.
"""

from fastapi import APIRouter, UploadFile, File, Form, status
from typing import Annotated

from ...dependencies import AbnormalServiceDep, RequestIdDep
from ...schemas.abnormal import (
    AbnormalResponse,
    BadConditionResponse,
)
from ...schemas.common import AbnormalType, AbnormalStatus

router = APIRouter(prefix="/abnormal", tags=["Abnormal Alerts"])


@router.post(
    "",
    response_model=AbnormalResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create abnormal alert",
    description="""
    Report an abnormal condition detected during train operation.

    **Use cases:**
    - Unusual vibrations detected
    - Temperature anomalies
    - Other operational alerts

    The photo file is required for visual verification.
    """,
    responses={
        201: {"description": "Alert created successfully"},
        400: {"description": "Invalid request data"},
        422: {"description": "Validation error"},
        502: {"description": "Upstream service error"},
    },
)
async def create_abnormal(
    service: AbnormalServiceDep,
    request_id: RequestIdDep,
    carbin_no: Annotated[str, Form(alias="carbinNo", description="车厢编号 (Carriage number)")],
    descr: Annotated[str, Form(description="告警信息 (Alert description)")],
    file: Annotated[UploadFile, File(description="Photo file")],
) -> AbnormalResponse:
    """Create an abnormal alert with photo evidence."""
    from ...schemas.abnormal import AbnormalCreateRequest

    request = AbnormalCreateRequest(carbin_no=carbin_no, descr=descr)
    response = await service.create_abnormal(request, file)
    response.request_id = request_id
    return response


@router.post(
    "/condition",
    response_model=BadConditionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Report carriage condition",
    description="""
    Report specific carriage anomalies like unlocked doors or foreign objects.

    **Abnormal Types:**
    - `1` (UNLOCKED): Lock not properly secured (未落锁)
    - `2` (FOREIGN_OBJECT): Foreign object detected (有异物)

    **Status:**
    - `0`: No abnormality detected (无异常)
    - `1`: Abnormality detected (有异常)

    Photo is optional but recommended for verification.
    """,
)
async def create_bad_condition(
    service: AbnormalServiceDep,
    request_id: RequestIdDep,
    carbin_no: Annotated[str, Form(alias="carbinNo", description="车厢编号")],
    abnormal_type: Annotated[AbnormalType, Form(alias="abnormalType", description="异常类型 (1=未落锁, 2=有异物)")],
    is_abnormal: Annotated[AbnormalStatus, Form(alias="isAbnormal", description="是否异常 (0=无异常, 1=有异常)")],
    descr: Annotated[str | None, Form(description="异常描述")] = None,
    file: Annotated[UploadFile | None, File(description="Optional photo")] = None,
) -> BadConditionResponse:
    """Report carriage condition (lock status, foreign objects)."""
    from ...schemas.abnormal import BadConditionCreateRequest

    request = BadConditionCreateRequest(
        carbin_no=carbin_no,
        abnormal_type=abnormal_type,
        is_abnormal=is_abnormal,
        descr=descr,
    )
    response = await service.create_bad_condition(request, file)
    response.request_id = request_id
    return response
