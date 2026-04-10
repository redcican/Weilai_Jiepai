"""
FastAPI Dependencies

Dependency injection for services and repositories.
"""

from typing import Annotated
from fastapi import Depends, Request

from .config import Settings, get_settings
from .repositories.dms import DMSRepository, get_dms_repository
from .services.abnormal import AbnormalService
from .services.ticket import TicketService
from .services.train_id import TrainIDService, get_train_id_service_singleton
from .services.signal_light import SignalLightService, get_signal_light_service_singleton
from .services.pedestrian import PedestrianService, get_pedestrian_service_singleton


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


async def get_ticket_service(
    repository: RepositoryDep,
) -> TicketService:
    """Get ticket service instance."""
    return TicketService(repository)


# Type aliases for dependency injection
AbnormalServiceDep = Annotated[AbnormalService, Depends(get_abnormal_service)]
TicketServiceDep = Annotated[TicketService, Depends(get_ticket_service)]


# Train ID service (standalone — no repository needed)
async def get_train_id_service() -> TrainIDService:
    """Get train ID service instance."""
    return get_train_id_service_singleton()


TrainIDServiceDep = Annotated[TrainIDService, Depends(get_train_id_service)]


# Signal Light service (standalone — no repository needed)
async def get_signal_light_service() -> SignalLightService:
    """Get signal light service instance."""
    return get_signal_light_service_singleton()


SignalLightServiceDep = Annotated[SignalLightService, Depends(get_signal_light_service)]


# Pedestrian Detection service (standalone — no repository needed)
async def get_pedestrian_service() -> PedestrianService:
    """Get pedestrian detection service instance."""
    return get_pedestrian_service_singleton()


PedestrianServiceDep = Annotated[PedestrianService, Depends(get_pedestrian_service)]


# Request ID dependency
def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, "request_id", "unknown")


RequestIdDep = Annotated[str, Depends(get_request_id)]
