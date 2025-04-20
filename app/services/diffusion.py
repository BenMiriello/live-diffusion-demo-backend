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

from streamdiffusion import StreamDiffusion
from streamdiffusion.schedulers import KarrasDiffusionSchedulers
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import torch

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
        """Load a StreamDiffusion model"""
        if self._loading:
            logger.warning("Already loading a model, please wait")
            return

        self._loading = True
        self._model_loaded = False

        try:
            logger.info(f"Loading model: {model_id}")

            # Use an event loop to run CPU-intensive tasks
            loop = asyncio.get_event_loop()

            # Load models on a separate thread to not block the main event loop
            def _load_model():
                # Load model components
                unet = UNet2DConditionModel.from_pretrained(
                    model_id, subfolder="unet", torch_dtype=torch.float16
                )
                tokenizer = CLIPTokenizer.from_pretrained(
                    model_id, subfolder="tokenizer"
                )
                text_encoder = CLIPTextModel.from_pretrained(
                    model_id, subfolder="text_encoder", torch_dtype=torch.float16
                )
                vae = AutoencoderKL.from_pretrained(
                    model_id, subfolder="vae", torch_dtype=torch.float16
                )

                # Create StreamDiffusion instance
                stream = StreamDiffusion(
                    unet=unet,
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    scheduler_type=KarrasDiffusionSchedulers.DPM_SOLVER_MULTISTEP,
                    device=torch.device(f"cuda:{settings.gpu_device}"),
                )

                # Configure StreamDiffusion
                stream.enable_vae_slicing()
                stream.enable_xformers_memory_efficient_attention()

                # Set inference parameters
                stream.update_tokenizer()
                stream.set_width(512)
                stream.set_height(512)

                return stream

            # Run model loading in a thread pool
            self._stream = await loop.run_in_executor(None, _load_model)

            # Set model details
            self._model_id = model_id
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

    async def process_frame(self, frame_data: bytes) -> Tuple[bytes, int, int]:
        """Process a frame with StreamDiffusion"""
        if not self._model_loaded:
            raise ValueError("Model not loaded")

        try:
            # Decode the input image
            image, width, height = decode_image(frame_data)

            # Ensure image is RGB
            if image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]

            # Resize to model dimensions if needed
            image = resize_image(image, width=512, height=512)

            # Convert to torch tensor
            torch_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
                device=f"cuda:{settings.gpu_device}", dtype=torch.float16
            )
            torch_image = torch_image / 127.5 - 1.0

            # Run diffusion
            loop = asyncio.get_event_loop()

            def _run_diffusion():
                # Set text prompt
                self._stream.set_text_prompt(
                    self._settings.prompt, self._settings.negative_prompt
                )

                # Set denoising strength
                self._stream.set_strength(self._settings.denoising_strength)

                # Process the image
                output = self._stream.stream_image(
                    torch_image, 
                    step=self._settings.steps,
                    cfg_scale=self._settings.guidance
                )

                # Convert back to numpy
                output_np = output.permute(0, 2, 3, 1).cpu().numpy()
                output_np = ((output_np[0] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

                return output_np

            # Run diffusion in a thread pool
            processed = await loop.run_in_executor(None, _run_diffusion)

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
