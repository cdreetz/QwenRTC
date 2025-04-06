# app/api/router.py
from fastapi import APIRouter
from app.api.endpoints import inference

# Create the main API router
router = APIRouter()

# Include all endpoint routers
router.include_router(inference.router, prefix="/inference", tags=["inference"])


