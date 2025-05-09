from typing import List
import os
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    log_level: str = Field("info", env="LOG_LEVEL")

    # CORS
    cors_origins: List[str] = Field(
        ["http://localhost:5173", "http://localhost:4173"],
        env="CORS_ORIGINS"
    )

    # StreamDiffusion Settings
    model_id: str = Field("stabilityai/sd-turbo", env="MODEL_ID")
    gpu_device: int = Field(0, env="GPU_DEVICE")
    denoising_strength: float = Field(0.5, env="DENOISING_STRENGTH")
    
    # StreamDiffusion Service URL
    streamdiffusion_url: str = Field("http://streamdiffusion:8001", env="STREAMDIFFUSION_URL")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "protected_namespaces": ("settings_",),
    }

# Create global settings object
settings = Settings(_env_file=".env")
