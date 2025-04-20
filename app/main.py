from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
import asyncio

from app.core.config import settings
from app.api import api_router
from app.api.websocket import handle_websocket
from app.services.diffusion import get_diffusion_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Live Diffusion Demo API",
    description="API for the Live Diffusion Demo with StreamDiffusion",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API router
app.include_router(api_router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting up Live Diffusion Demo API")
    
    # Pre-load the default model if specified
    diffusion_service = get_diffusion_service()
    if settings.model_id:
        try:
            logger.info(f"Pre-loading model: {settings.model_id}")
            # Load the model in a background task
            asyncio.create_task(diffusion_service.load_model(settings.model_id))
        except Exception as e:
            logger.error(f"Error pre-loading model: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Live Diffusion Demo API")
    # Add any cleanup here


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to Live Diffusion Demo API"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "api_version": app.version,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await handle_websocket(websocket)
