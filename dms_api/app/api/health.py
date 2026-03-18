"""
Health Check Endpoints

Kubernetes-ready health and readiness probes.
"""

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from datetime import datetime

from ..config import Settings, get_settings
from ..repositories.dms import get_dms_repository, DMSRepository
from ..schemas.common import HealthStatus

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health check response."""

    status: HealthStatus = Field(..., description="Overall health status")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment name")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    checks: dict[str, dict] = Field(default_factory=dict, description="Individual check results")


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    ready: bool = Field(..., description="Whether the service is ready")
    checks: dict[str, bool] = Field(default_factory=dict)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Basic health check for load balancers and monitoring.",
)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """
    Health check endpoint.

    Returns basic health information about the service.
    Used by load balancers and monitoring systems.
    """
    return HealthResponse(
        status=HealthStatus.HEALTHY,
        version=settings.app_version,
        environment=settings.environment,
        checks={
            "api": {"status": "ok"},
        },
    )


@router.get(
    "/health/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Kubernetes liveness probe. Returns 200 if the service is alive.",
)
async def liveness_probe() -> dict:
    """
    Liveness probe for Kubernetes.

    Simply returns 200 to indicate the service is running.
    """
    return {"status": "alive"}


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Kubernetes readiness probe. Checks if the service is ready to accept traffic.",
)
async def readiness_probe(
    settings: Settings = Depends(get_settings),
) -> ReadinessResponse:
    """
    Readiness probe for Kubernetes.

    Checks if all dependencies are available and the service
    is ready to accept traffic.
    """
    checks = {}

    # Check DMS backend
    try:
        repository = await get_dms_repository(settings)
        dms_healthy = await repository.health_check()
        checks["dms_backend"] = dms_healthy
    except Exception:
        checks["dms_backend"] = False

    # Overall readiness
    ready = all(checks.values()) if checks else True

    return ReadinessResponse(ready=ready, checks=checks)


@router.get(
    "/health/detailed",
    response_model=HealthResponse,
    summary="Detailed health check",
    description="Detailed health check with dependency status.",
)
async def detailed_health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """
    Detailed health check with all dependency statuses.

    Useful for debugging and monitoring dashboards.
    """
    checks = {}
    overall_status = HealthStatus.HEALTHY

    # Check DMS backend
    try:
        repository = await get_dms_repository(settings)
        dms_healthy = await repository.health_check()
        checks["dms_backend"] = {
            "status": "ok" if dms_healthy else "degraded",
            "url": str(settings.dms_base_url),
        }
        if not dms_healthy:
            overall_status = HealthStatus.DEGRADED
    except Exception as e:
        checks["dms_backend"] = {
            "status": "error",
            "error": str(e),
        }
        overall_status = HealthStatus.DEGRADED

    # Configuration check
    checks["configuration"] = {
        "status": "ok",
        "environment": settings.environment,
        "debug": settings.debug,
    }

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        checks=checks,
    )
