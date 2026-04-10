"""
API v1 Router

Main router that combines all v1 API endpoints.
"""

from fastapi import APIRouter

from .abnormal import router as abnormal_router
from .ticket import router as ticket_router
from .train_id import router as train_id_router
from .signal_light import router as signal_light_router
from .pedestrian import router as pedestrian_router

router = APIRouter(prefix="/api/v1")

# Include all sub-routers
router.include_router(abnormal_router)
router.include_router(ticket_router)
router.include_router(train_id_router)
router.include_router(signal_light_router)
router.include_router(pedestrian_router)
