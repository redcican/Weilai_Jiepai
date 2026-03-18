"""
Abnormal Alert Service

Business logic for abnormal alert and bad condition reporting.
"""

from fastapi import UploadFile

from .base import BaseService
from ..repositories.dms import DMSRepository
from ..schemas.abnormal import (
    AbnormalCreateRequest,
    AbnormalResponse,
    AbnormalData,
    BadConditionCreateRequest,
    BadConditionResponse,
    BadConditionData,
)
from ..schemas.common import ResponseCode
from ..core.exceptions import DMSUpstreamError, DMSFileError


class AbnormalService(BaseService):
    """Service for abnormal alert operations."""

    def __init__(self, repository: DMSRepository):
        super().__init__(repository)

    async def create_abnormal(
        self,
        request: AbnormalCreateRequest,
        file: UploadFile,
    ) -> AbnormalResponse:
        """
        Create an abnormal alert.

        Args:
            request: Alert request data
            file: Photo file

        Returns:
            AbnormalResponse with result
        """
        # Validate file
        if not file.filename:
            raise DMSFileError(message="Filename is required")

        content_type = file.content_type or "application/octet-stream"
        file_content = await file.read()

        if not file_content:
            raise DMSFileError(message="File is empty", filename=file.filename)

        self._logger.info(
            f"Creating abnormal alert: carbin_no={request.carbin_no}",
            extra={"extra_fields": {"carbin_no": request.carbin_no}},
        )

        # Call repository
        result = await self._repository.save_abnormal(
            carbin_no=request.carbin_no,
            descr=request.descr,
            file_content=file_content,
            filename=file.filename,
            content_type=content_type,
        )

        # Build response
        if result.is_success:
            data = AbnormalData(
                carbin_no=request.carbin_no,
                descr=request.descr,
            )
            return AbnormalResponse.ok(data=data, message=result.msg or "Alert created")

        # Handle upstream error
        raise DMSUpstreamError(
            upstream_code=result.code,
            upstream_message=result.msg,
            message=f"Failed to create abnormal alert: {result.msg}",
        )

    async def create_bad_condition(
        self,
        request: BadConditionCreateRequest,
        file: UploadFile | None = None,
    ) -> BadConditionResponse:
        """
        Report bad carriage condition.

        Args:
            request: Condition report data
            file: Optional photo file

        Returns:
            BadConditionResponse with result
        """
        self._logger.info(
            f"Reporting bad condition: carbin_no={request.carbin_no}, "
            f"type={request.abnormal_type}, is_abnormal={request.is_abnormal}",
        )

        # Prepare file if provided
        file_content = None
        filename = None
        content_type = None

        if file:
            file_content = await file.read()
            filename = file.filename or "photo.jpg"
            content_type = file.content_type or "application/octet-stream"

        # Call repository
        result = await self._repository.save_bad_condition(
            carbin_no=request.carbin_no,
            abnormal_type=request.abnormal_type.value,
            is_abnormal=request.is_abnormal.value,
            descr=request.descr,
            file_content=file_content,
            filename=filename,
            content_type=content_type,
        )

        if result.is_success:
            data = BadConditionData(
                carbin_no=request.carbin_no,
                abnormal_type=request.abnormal_type,
                is_abnormal=request.is_abnormal,
                descr=request.descr,
            )
            return BadConditionResponse.ok(data=data, message=result.msg or "Condition reported")

        raise DMSUpstreamError(
            upstream_code=result.code,
            upstream_message=result.msg,
            message=f"Failed to report bad condition: {result.msg}",
        )
