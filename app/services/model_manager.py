import logging
from typing import List, Dict, Optional

from app.models.models import ModelInfo

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages available diffusion models
    
    This is a placeholder that will be expanded with actual model management.
    """

    def __init__(self):
        """Initialize the model manager"""
        self._models = [
            ModelInfo(
                id="stabilityai/sd-turbo",
                name="Stable Diffusion Turbo",
                description="Fast and efficient Stable Diffusion model optimized for real-time generation",
                is_loaded=False,
            ),
            ModelInfo(
                id="runwayml/stable-diffusion-v1-5",
                name="Stable Diffusion 1.5",
                description="Stable Diffusion v1.5 for high-quality image generation",
                is_loaded=False,
            ),
            ModelInfo(
                id="stabilityai/sdxl-turbo",
                name="SDXL Turbo",
                description="Fast SDXL model for high-resolution real-time generation",
                is_loaded=False,
            ),
        ]
        self._models_dict = {model.id: model for model in self._models}

    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models"""
        return self._models

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        return self._models_dict.get(model_id)

    def set_model_loaded(self, model_id: str, loaded: bool = True) -> None:
        """Set a model's loaded status"""
        # Update the model in our list
        if model_id in self._models_dict:
            self._models_dict[model_id].is_loaded = loaded
            
            # If we're marking this as loaded, mark all others as unloaded
            if loaded:
                for other_id, model in self._models_dict.items():
                    if other_id != model_id:
                        model.is_loaded = False
        else:
            logger.warning(f"Model {model_id} not found")


# Create a global instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
