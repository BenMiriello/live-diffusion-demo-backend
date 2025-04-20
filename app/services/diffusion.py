import asyncio
import logging
import time
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import io
from PIL import Image, ImageDraw, ImageFont
import threading
import random
import os

from app.models.settings import DiffusionSettings
from app.utils.image import decode_image, encode_image, resize_image
from app.core.config import settings

logger = logging.getLogger(__name__)


class DiffusionService:
    """
    Service for handling diffusion model operations
    
    This is a placeholder implementation that will be replaced with actual
    StreamDiffusion integration.
    """

    def __init__(self):
        """Initialize the diffusion service"""
        self._settings = DiffusionSettings()
        self._model_loaded = False
        self._model_id = None
        self._model_name = None
        self._lock = threading.Lock()
        self._loading = False

    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model_loaded

    def get_model_id(self) -> Optional[str]:
        """Get current model ID"""
        return self._model_id

    def get_model_name(self) -> Optional[str]:
        """Get current model name"""
        return self._model_name

    def get_settings(self) -> DiffusionSettings:
        """Get current settings"""
        return self._settings

    def update_settings(self, settings_update: Dict[str, Any]) -> None:
        """Update settings with partial update"""
        # Update only the fields that were provided
        for key, value in settings_update.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
            else:
                logger.warning(f"Unknown setting: {key}")

    async def load_model(self, model_id: str) -> None:
        """
        Load a diffusion model
        
        This is a placeholder that simulates loading a model.
        """
        if self._loading:
            logger.warning("Already loading a model, please wait")
            return

        self._loading = True
        self._model_loaded = False

        try:
            logger.info(f"Loading model: {model_id}")
            
            # Simulate model loading time
            await asyncio.sleep(2)
            
            # Set model details
            self._model_id = model_id
            
            if model_id == "stabilityai/sd-turbo":
                self._model_name = "Stable Diffusion Turbo"
            else:
                self._model_name = model_id.split("/")[-1]
            
            logger.info(f"Model loaded: {self._model_name}")
            self._model_loaded = True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._model_loaded = False
            self._model_id = None
            self._model_name = None
            raise
        
        finally:
            self._loading = False

    async def process_frame(
        self, frame_data: bytes
    ) -> Tuple[bytes, int, int]:
        """
        Process a frame with the diffusion model
        
        This is a placeholder that creates a mock processed frame.
        
        Args:
            frame_data: Binary image data
            
        Returns:
            Tuple of (processed binary image data, width, height)
        """
        try:
            # Decode the input image
            image, width, height = decode_image(frame_data)
            
            # In the real implementation, this would send the frame to StreamDiffusion
            # For now, we'll create a simple mock result
            
            # Create a blank image with the prompt text
            processed = image.copy()
            
            # Add a semi-transparent overlay
            overlay = np.zeros_like(processed)
            overlay[:] = (0, 0, 0, 128)
            alpha = 0.5
            processed = cv2.addWeighted(processed, 1 - alpha, overlay, alpha, 0)
            
            # Convert to PIL Image to add text
            pil_image = Image.fromarray(processed)
            draw = ImageDraw.Draw(pil_image)
            
            # Add the prompt text
            prompt = self._settings.prompt or "No prompt provided"
            draw.text((10, 10), f"Prompt: {prompt}", fill=(255, 255, 255))
            draw.text((10, 40), f"Denoising: {self._settings.denoising_strength}", fill=(255, 255, 255))
            draw.text((10, 70), f"Timestamp: {time.time()}", fill=(255, 255, 255))
            
            # Convert back to numpy
            processed = np.array(pil_image)
            
            # Encode the result
            result_data, width, height = encode_image(processed)
            
            return result_data, width, height
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            raise


# Create a global instance
_diffusion_service = None


def get_diffusion_service() -> DiffusionService:
    """Get the global diffusion service instance"""
    global _diffusion_service
    if _diffusion_service is None:
        _diffusion_service = DiffusionService()
    return _diffusion_service
