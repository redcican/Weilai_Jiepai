"""
API v1 Router

Main router that combines all v1 API endpoints.
"""

from fastapi import APIRouter

from .abnormal import router as abnormal_router
from .container import router as container_router
from .signal import router as signal_router
from .ticket import router as ticket_router
from .train_id import router as train_id_router

router = APIRouter(prefix="/api/v1")

# Include all sub-routers
router.include_router(abnormal_router)
router.include_router(container_router)
router.include_router(signal_router)
router.include_router(ticket_router)
router.include_router(train_id_router)
