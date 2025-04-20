from fastapi import WebSocket
import json
import logging
from typing import Dict, Set, Any, List, Optional
import asyncio

from app.models.websocket import (
    WebSocketMessage,
    StatusMessage,
    ErrorMessage,
    MessageType,
)

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.processing_frames: Set[str] = set()

    async def connect(self, websocket: WebSocket) -> str:
        """Connect a client and return the connection ID"""
        await websocket.accept()
        
        # Generate a unique ID for this connection
        connection_id = f"conn_{id(websocket)}"
        self.active_connections[connection_id] = websocket
        
        logger.info(f"Client connected: {connection_id}")
        return connection_id

    def disconnect(self, connection_id: str) -> None:
        """Disconnect a client by ID"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            logger.info(f"Client disconnected: {connection_id}")
        
        # Remove any processing frames for this connection
        if connection_id in self.processing_frames:
            self.processing_frames.remove(connection_id)

    async def send_message(
        self, connection_id: str, message: WebSocketMessage
    ) -> None:
        """Send a JSON message to a specific client"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_json(message.dict())
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                # Try to clean up if we can't send
                await self.disconnect_websocket(connection_id)

    async def send_binary(self, connection_id: str, data: bytes) -> None:
        """Send binary data to a specific client"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_bytes(data)
            except Exception as e:
                logger.error(f"Error sending binary data to {connection_id}: {e}")
                # Try to clean up if we can't send
                await self.disconnect_websocket(connection_id)

    async def send_error(
        self, connection_id: str, error: str, details: Optional[str] = None
    ) -> None:
        """Send an error message to a specific client"""
        message = ErrorMessage(error=error, details=details)
        await self.send_message(connection_id, message)

    async def broadcast_message(self, message: WebSocketMessage) -> None:
        """Send a message to all connected clients"""
        for connection_id in list(self.active_connections.keys()):
            await self.send_message(connection_id, message)

    async def disconnect_websocket(self, connection_id: str) -> None:
        """Forcibly disconnect a websocket"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.close()
            except Exception:
                pass  # Already closed or error, just ignore
            finally:
                self.disconnect(connection_id)

    def mark_processing(self, connection_id: str) -> None:
        """Mark a connection as currently processing a frame"""
        self.processing_frames.add(connection_id)

    def mark_done_processing(self, connection_id: str) -> None:
        """Mark a connection as done processing"""
        if connection_id in self.processing_frames:
            self.processing_frames.remove(connection_id)

    def is_processing(self, connection_id: str) -> bool:
        """Check if a connection is currently processing a frame"""
        return connection_id in self.processing_frames

    def get_active_connections_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)


# Create a global instance
connection_manager = ConnectionManager()
