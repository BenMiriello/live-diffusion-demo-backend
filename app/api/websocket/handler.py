import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, Union
import io
import base64
from fastapi import WebSocket, WebSocketDisconnect

from app.api.websocket.connection import connection_manager
from app.models.websocket import (
    MessageType,
    WebSocketMessage,
    FrameMessage,
    StatusMessage,
    ProcessedFrameMessage,
    ErrorMessage,
)
from app.services.diffusion import DiffusionService, get_diffusion_service
from app.models.settings import DiffusionSettings

logger = logging.getLogger(__name__)


async def handle_websocket(websocket: WebSocket):
    """Main WebSocket handler that routes messages to appropriate handlers"""
    # Accept the connection
    connection_id = await connection_manager.connect(websocket)
    
    # Initialize status variables
    diffusion_service = get_diffusion_service()
    model_loaded = diffusion_service.is_model_loaded()
    model_name = diffusion_service.get_model_name() if model_loaded else None
    
    # Send initial status
    await connection_manager.send_message(
        connection_id,
        StatusMessage(
            connected=True,
            processing=False,
            model_loaded=model_loaded,
            model_name=model_name,
            message="Connected to server"
        )
    )

    try:
        while True:
            # This will block until a message is received
            message = await websocket.receive()
            
            # Handle different message types (text vs binary)
            if "text" in message:
                try:
                    # Parse as JSON
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "unknown")
                    
                    if msg_type == MessageType.SETTINGS:
                        await handle_settings_message(connection_id, data)
                    elif msg_type == MessageType.PING:
                        await handle_ping_message(connection_id)
                    else:
                        logger.warning(f"Unknown message type: {msg_type}")
                        await connection_manager.send_error(
                            connection_id, f"Unknown message type: {msg_type}"
                        )
                except json.JSONDecodeError:
                    logger.warning("Received malformed JSON")
                    await connection_manager.send_error(
                        connection_id, "Malformed JSON message"
                    )
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await connection_manager.send_error(
                        connection_id, f"Error processing message: {str(e)}"
                    )
            
            elif "bytes" in message:
                # Handle binary data (frames)
                if connection_manager.is_processing(connection_id):
                    # Skip processing if we're already processing a frame
                    logger.debug(f"Skipping frame, already processing for {connection_id}")
                    continue
                
                try:
                    # Mark as processing
                    connection_manager.mark_processing(connection_id)
                    
                    # Process the frame
                    await handle_frame(connection_id, message["bytes"])
                    
                    # Mark as done processing
                    connection_manager.mark_done_processing(connection_id)
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    connection_manager.mark_done_processing(connection_id)
                    await connection_manager.send_error(
                        connection_id, f"Error processing frame: {str(e)}"
                    )
    
    except WebSocketDisconnect:
        # Client disconnected
        logger.info(f"Client disconnected: {connection_id}")
    except Exception as e:
        # Unexpected error
        logger.error(f"WebSocket error for {connection_id}: {e}")
    finally:
        # Always clean up the connection
        connection_manager.disconnect(connection_id)


async def handle_ping_message(connection_id: str):
    """Handle ping message"""
    await connection_manager.send_message(
        connection_id,
        WebSocketMessage(type=MessageType.PONG, data={"timestamp": time.time()})
    )


async def handle_settings_message(connection_id: str, data: Dict[str, Any]):
    """Handle settings update message"""
    settings_data = data.get("data", {})
    if not settings_data:
        await connection_manager.send_error(
            connection_id, "Missing settings data"
        )
        return
    
    try:
        # Update settings in diffusion service
        diffusion_service = get_diffusion_service()
        diffusion_service.update_settings(settings_data)
        
        # Send back confirmation
        await connection_manager.send_message(
            connection_id,
            WebSocketMessage(
                type=MessageType.SETTINGS,
                data={"settings": diffusion_service.get_settings().dict()}
            )
        )
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        await connection_manager.send_error(
            connection_id, f"Error updating settings: {str(e)}"
        )


async def handle_frame(connection_id: str, frame_data: bytes):
    """Handle incoming video frame"""
    start_time = time.time()
    
    try:
        # Get the diffusion service
        diffusion_service = get_diffusion_service()
        
        # Skip processing if model isn't loaded
        if not diffusion_service.is_model_loaded():
            await connection_manager.send_error(
                connection_id, "Model not loaded", "Please wait for the model to load"
            )
            return
        
        # Process the frame with StreamDiffusion
        # This is a placeholder until we implement the actual StreamDiffusion integration
        processed_frame, width, height = await diffusion_service.process_frame(frame_data)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Send frame metadata
        await connection_manager.send_message(
            connection_id,
            ProcessedFrameMessage(
                width=width,
                height=height,
                timestamp=time.time(),
                processing_time=processing_time
            )
        )
        
        # Send the binary frame data
        await connection_manager.send_binary(connection_id, processed_frame)
        
        logger.debug(
            f"Frame processed in {processing_time:.2f}s for {connection_id}"
        )
    
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        await connection_manager.send_error(
            connection_id, f"Error processing frame: {str(e)}"
        )
