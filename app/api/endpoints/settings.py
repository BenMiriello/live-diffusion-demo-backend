from fastapi import APIRouter, HTTPException, status
from typing import Any, Dict

from app.models.settings import (
    DiffusionSettings,
    SettingsResponse,
    SettingsUpdateRequest,
)
from app.services.diffusion import get_diffusion_service

router = APIRouter()


@router.get(
    "",
    response_model=SettingsResponse,
    summary="Get current diffusion settings",
    description="Returns the current settings used for diffusion processing",
)
async def get_settings() -> SettingsResponse:
    """Get current diffusion settings"""
    diffusion_service = get_diffusion_service()
    settings = diffusion_service.get_settings()
    return SettingsResponse(settings=settings)


@router.put(
    "",
    response_model=SettingsResponse,
    summary="Update diffusion settings",
    description="Update the settings used for diffusion processing",
)
async def update_settings(update: SettingsUpdateRequest) -> SettingsResponse:
    """Update diffusion settings"""
    diffusion_service = get_diffusion_service()
    
    # Update only the fields that were provided
    update_dict = {k: v for k, v in update.dict().items() if v is not None}
    
    try:
        diffusion_service.update_settings(update_dict)
        return SettingsResponse(settings=diffusion_service.get_settings())
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating settings: {str(e)}"
        )
