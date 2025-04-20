import asyncio
import logging
import time
import base64
import httpx
from typing import Dict, Any, Tuple, Optional, List
import json
import io
import os
import numpy as np
from PIL import Image

from app.models.settings import DiffusionSettings
from app.utils.image import decode_image, encode_image, resize_image
from app.core.config import settings

logger = logging.getLogger(__name__)

class DiffusionService:
    """
    Service for handling diffusion model operations through an external StreamDiffusion service
    """

    def __init__(self):
        """Initialize the diffusion service"""
        self._settings = DiffusionSettings()
        self._model_loaded = False
        self._model_id = None
        self._model_name = None
        self._stream_service_url = os.environ.get("STREAMDIFFUSION_URL", "http://localhost:8001")
        self._client = httpx.AsyncClient(timeout=60.0)  # Longer timeout for image processing

    async def _check_service_status(self) -> Dict[str, Any]:
        """Check the status of the StreamDiffusion service"""
        try:
            response = await self._client.get(f"{self._stream_service_url}/status")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error checking service status: {response.status_code} {response.text}")
                return {"status": "error", "model_loaded": False}
        except Exception as e:
            logger.error(f"Error connecting to StreamDiffusion service: {e}")
            return {"status": "error", "model_loaded": False}

    async def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        status = await self._check_service_status()
        self._model_loaded = status.get("model_loaded", False)
        self._model_id = status.get("model_id")
        if self._model_id:
            self._model_name = self._model_id.split("/")[-1]
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
        """Load a StreamDiffusion model via the service"""
        try:
            logger.info(f"Loading model: {model_id}")
            
            # Call the service to load the model
            response = await self._client.post(
                f"{self._stream_service_url}/load",
                json={"model_id": model_id}
            )
            
            if response.status_code == 200:
                self._model_id = model_id
                self._model_name = model_id.split("/")[-1]
                self._model_loaded = True
                logger.info(f"Model loaded: {self._model_name}")
            else:
                logger.error(f"Error loading model: {response.status_code} {response.text}")
                self._model_loaded = False
                self._model_id = None
                self._model_name = None
                raise ValueError(f"Failed to load model: {response.text}")
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._model_loaded = False
            self._model_id = None
            self._model_name = None
            raise

    async def process_frame(self, frame_data: bytes) -> Tuple[bytes, int, int]:
        """Process a frame with StreamDiffusion service"""
        if not await self.is_model_loaded():
            raise ValueError("Model not loaded")

        try:
            # Decode the input image to get dimensions
            image, width, height = decode_image(frame_data)
            
            # Encode the frame data to base64
            b64_image = base64.b64encode(frame_data).decode("utf-8")
            
            # Prepare the request data
            request_data = {
                "image": b64_image,
                "prompt": self._settings.prompt,
                "negative_prompt": self._settings.negative_prompt,
                "denoising_strength": self._settings.denoising_strength,
                "guidance_scale": self._settings.guidance,
                "steps": self._settings.steps,
                "width": 512,  # Fixed size for now, could be made configurable
                "height": 512,
            }
            
            # Send the request to the service
            start_time = time.time()
            response = await self._client.post(
                f"{self._stream_service_url}/process/base64",
                json=request_data
            )
            
            if response.status_code != 200:
                logger.error(f"Error processing image: {response.status_code} {response.text}")
                raise ValueError(f"Error from StreamDiffusion service: {response.text}")
            
            # Parse the response
            result = response.json()
            processing_time = time.time() - start_time
            logger.info(f"Image processed in {processing_time:.2f}s")
            
            # Decode the processed image
            processed_b64 = result.get("processed_image")
            if not processed_b64:
                raise ValueError("No processed image in response")
            
            processed_data = base64.b64decode(processed_b64)
            
            # Get image dimensions
            img = Image.open(io.BytesIO(processed_data))
            width, height = img.size
            
            return processed_data, width, height

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
