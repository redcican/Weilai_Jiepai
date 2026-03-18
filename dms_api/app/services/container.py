"""
Container Service

Business logic for container identification operations.
"""

from fastapi import UploadFile

from .base import BaseService
from ..repositories.dms import DMSRepository
from ..schemas.container import (
    ContainerCreateRequest,
    ContainerResponse,
    ContainerData,
    ContainerBatchItem,
)
from ..schemas.common import BatchResponse, BatchItemResult
from ..core.exceptions import DMSUpstreamError, DMSFileError


class ContainerService(BaseService):
    """Service for container identification operations."""

    def __init__(self, repository: DMSRepository):
        super().__init__(repository)

    async def create_container(
        self,
        request: ContainerCreateRequest,
        file: UploadFile,
    ) -> ContainerResponse:
        """
        Report container identification.

        Args:
            request: Container data
            file: Photo file

        Returns:
            ContainerResponse with result
        """
        # Validate file
        if not file.filename:
            raise DMSFileError(message="Filename is required")

        content_type = file.content_type or "application/octet-stream"
        file_content = await file.read()

        if not file_content:
            raise DMSFileError(message="File is empty", filename=file.filename)

        self._logger.info(
            f"Creating container record: train={request.train_no}, "
            f"carbin={request.carbin_no}, container={request.container_no}",
        )

        # Call repository
        result = await self._repository.save_container(
            train_no=request.train_no,
            carbin_no=request.carbin_no,
            container_no=request.container_no,
            file_content=file_content,
            filename=file.filename,
            content_type=content_type,
        )

        if result.is_success:
            data = ContainerData(
                train_no=request.train_no,
                carbin_no=request.carbin_no,
                container_no=request.container_no,
            )
            return ContainerResponse.ok(data=data, message=result.msg or "Container recorded")

        raise DMSUpstreamError(
            upstream_code=result.code,
            upstream_message=result.msg,
            message=f"Failed to record container: {result.msg}",
        )

    async def batch_create_containers(
        self,
        items: list[ContainerBatchItem],
        files: list[UploadFile],
    ) -> BatchResponse:
        """
        Batch create container records.

        Args:
            items: List of container data
            files: List of photo files (same order as items)

        Returns:
            BatchResponse with results
        """
        if len(items) != len(files):
            raise DMSFileError(
                message=f"Item count ({len(items)}) doesn't match file count ({len(files)})"
            )

        results: list[BatchItemResult] = []
        succeeded = 0
        failed = 0

        for i, (item, file) in enumerate(zip(items, files)):
            try:
                request = ContainerCreateRequest(
                    train_no=item.train_no,
                    carbin_no=item.carbin_no,
                    container_no=item.container_no,
                )
                response = await self.create_container(request, file)

                if response.success:
                    succeeded += 1
                    results.append(BatchItemResult(
                        index=i,
                        success=True,
                        data=response.data.model_dump() if response.data else None,
                    ))
                else:
                    failed += 1
                    results.append(BatchItemResult(
                        index=i,
                        success=False,
                        error=response.message,
                    ))

            except Exception as e:
                failed += 1
                results.append(BatchItemResult(
                    index=i,
                    success=False,
                    error=str(e),
                ))

        return BatchResponse(
            total=len(items),
            succeeded=succeeded,
            failed=failed,
            results=results,
        )
