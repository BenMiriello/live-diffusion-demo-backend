from fastapi import APIRouter
from app.api.endpoints import settings, models, status

# Create the main API router
api_router = APIRouter()

# Include the endpoint routers
api_router.include_router(
    settings.router, prefix="/settings", tags=["settings"]
)
api_router.include_router(
    models.router, prefix="/models", tags=["models"]
)
api_router.include_router(
    status.router, prefix="/status", tags=["status"]
)
