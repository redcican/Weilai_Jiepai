"""
FastAPI Dependencies

Dependency injection for services and repositories.
"""

from typing import Annotated
from fastapi import Depends, Request

from .config import Settings, get_settings
from .repositories.dms import DMSRepository, get_dms_repository
from .services.abnormal import AbnormalService
from .services.container import ContainerService
from .services.signal import SignalService
from .services.ticket import TicketService


# Settings dependency
SettingsDep = Annotated[Settings, Depends(get_settings)]


# Repository dependencies
async def get_repository(
    settings: SettingsDep,
) -> DMSRepository:
    """Get DMS repository instance."""
    return await get_dms_repository(settings)


RepositoryDep = Annotated[DMSRepository, Depends(get_repository)]


# Service dependencies
async def get_abnormal_service(
    repository: RepositoryDep,
) -> AbnormalService:
    """Get abnormal service instance."""
    return AbnormalService(repository)


async def get_container_service(
    repository: RepositoryDep,
) -> ContainerService:
    """Get container service instance."""
    return ContainerService(repository)


async def get_signal_service(
    repository: RepositoryDep,
) -> SignalService:
    """Get signal service instance."""
    return SignalService(repository)


async def get_ticket_service(
    repository: RepositoryDep,
) -> TicketService:
    """Get ticket service instance."""
    return TicketService(repository)


# Type aliases for dependency injection
AbnormalServiceDep = Annotated[AbnormalService, Depends(get_abnormal_service)]
ContainerServiceDep = Annotated[ContainerService, Depends(get_container_service)]
SignalServiceDep = Annotated[SignalService, Depends(get_signal_service)]
TicketServiceDep = Annotated[TicketService, Depends(get_ticket_service)]


# Request ID dependency
def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, "request_id", "unknown")


RequestIdDep = Annotated[str, Depends(get_request_id)]
