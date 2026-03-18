"""
Signal Light Service

Business logic for signal light state change reporting.
"""

from fastapi import UploadFile

from .base import BaseService
from ..repositories.dms import DMSRepository
from ..schemas.signal import SignalChangeResponse, SignalChangeData
from ..core.exceptions import DMSUpstreamError, DMSFileError


class SignalService(BaseService):
    """Service for signal light operations."""

    def __init__(self, repository: DMSRepository):
        super().__init__(repository)

    async def report_signal_change(
        self,
        file: UploadFile,
    ) -> SignalChangeResponse:
        """
        Report signal light state change.

        Args:
            file: Signal light photo

        Returns:
            SignalChangeResponse with result
        """
        # Validate file
        if not file.filename:
            raise DMSFileError(message="Filename is required")

        content_type = file.content_type or "application/octet-stream"
        file_content = await file.read()

        if not file_content:
            raise DMSFileError(message="File is empty", filename=file.filename)

        self._logger.info(
            f"Reporting signal change: filename={file.filename}, size={len(file_content)}",
        )

        # Call repository
        result = await self._repository.save_signal_change(
            file_content=file_content,
            filename=file.filename,
            content_type=content_type,
        )

        if result.is_success:
            data = SignalChangeData(status="processed")
            return SignalChangeResponse.ok(data=data, message=result.msg or "Signal change recorded")

        raise DMSUpstreamError(
            upstream_code=result.code,
            upstream_message=result.msg,
            message=f"Failed to record signal change: {result.msg}",
        )
