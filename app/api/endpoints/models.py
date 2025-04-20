from fastapi import APIRouter, HTTPException, status
from typing import List

from app.models.models import (
    ModelInfo,
    ModelsResponse,
    ModelLoadRequest,
)
from app.services.diffusion import get_diffusion_service
from app.services.model_manager import get_model_manager

router = APIRouter()


@router.get(
    "",
    response_model=ModelsResponse,
    summary="Get available models",
    description="Returns list of available diffusion models and which one is currently loaded",
)
async def get_models() -> ModelsResponse:
    """Get available models and current model"""
    model_manager = get_model_manager()
    diffusion_service = get_diffusion_service()
    
    models = model_manager.get_available_models()
    current_model = diffusion_service.get_model_id()
    
    return ModelsResponse(
        models=models,
        current_model=current_model
    )


@router.post(
    "/load",
    response_model=ModelInfo,
    summary="Load a model",
    description="Loads a specified model for use in diffusion processing",
)
async def load_model(request: ModelLoadRequest) -> ModelInfo:
    """Load a specific model for use"""
    model_manager = get_model_manager()
    diffusion_service = get_diffusion_service()
    
    # Check if model exists
    available_models = model_manager.get_available_models()
    model_exists = any(model.id == request.model_id for model in available_models)
    
    if not model_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model_id}' not found"
        )
    
    try:
        # Load the model (async)
        await diffusion_service.load_model(request.model_id)
        
        # Get the model info of the loaded model
        model_info = next(
            (model for model in available_models if model.id == request.model_id),
            None
        )
        
        # Mark as loaded
        model_info.is_loaded = True
        
        return model_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}"
        )
