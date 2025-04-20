from fastapi import APIRouter
import time
import os
import psutil
import torch
import platform
from typing import Dict, Any, List

from app.core.config import settings
from app.services.diffusion import get_diffusion_service
from app.api.websocket.connection import connection_manager

# Get start time for uptime calculation
START_TIME = time.time()

router = APIRouter()


@router.get(
    "",
    summary="Get API status",
    description="Returns status information about the API and GPU",
)
async def get_status() -> Dict[str, Any]:
    """Get API status information"""
    diffusion_service = get_diffusion_service()
    
    # Calculate uptime
    uptime = time.time() - START_TIME
    
    # Get active connections
    active_connections = connection_manager.get_active_connections_count()
    
    # Get GPU info if available
    gpu_info = []
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            try:
                name = torch.cuda.get_device_name(i)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)  # MB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)  # MB
                
                gpu_info.append({
                    "index": i,
                    "name": name,
                    "memory_allocated_mb": round(memory_allocated, 2),
                    "memory_reserved_mb": round(memory_reserved, 2),
                })
            except Exception as e:
                gpu_info.append({
                    "index": i,
                    "error": str(e)
                })
    
    # Get system info
    process = psutil.Process(os.getpid())
    system_info = {
        "cpu_percent": process.cpu_percent(),
        "memory_percent": process.memory_percent(),
        "memory_mb": round(process.memory_info().rss / (1024 * 1024), 2),
        "threads": process.num_threads(),
        "python_version": platform.python_version(),
    }
    
    return {
        "status": "ok",
        "version": "0.1.0",
        "uptime_seconds": round(uptime, 2),
        "active_connections": active_connections,
        "model_loaded": diffusion_service.is_model_loaded(),
        "current_model": diffusion_service.get_model_name(),
        "cuda_available": cuda_available,
        "gpu_info": gpu_info,
        "system_info": system_info,
    }
