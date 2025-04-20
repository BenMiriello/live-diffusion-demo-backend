# Live Diffusion Demo - Backend

This directory contains the FastAPI backend for the Live Diffusion Demo, which processes webcam frames in real-time using StreamDiffusion.

## Setup Instructions

### Basic Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the example environment variables:
   ```bash
   cp .env.example .env
   ```

4. Start the server:
   ```bash
   python run.py
   ```

The server will be available at http://localhost:8000 by default.

## StreamDiffusion Integration

To integrate StreamDiffusion with this backend, you'll need to complete the following steps on your Debian Linux machine with an RTX 3090:

### 1. Install CUDA Dependencies

First, ensure your system has the correct CUDA drivers installed:

```bash
# Verify NVIDIA drivers are installed
nvidia-smi

# Install CUDA Toolkit 11.8 (if not already installed)
# Follow NVIDIA's instructions for Debian
```

### 2. Set Up Python Environment with Proper PyTorch

```bash
# Install PyTorch with CUDA support
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install StreamDiffusion

```bash
# Install StreamDiffusion
pip install streamdiffusion
```

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## WebSocket Protocol

The backend uses WebSockets for real-time communication with the frontend. Here's how the protocol works:

### Connection

Connect to the WebSocket endpoint at `ws://localhost:8000/ws`.

### Message Types

Messages are sent as either JSON (for control messages) or binary data (for image frames).

#### JSON Messages

JSON messages have the following structure:
```json
{
  "type": "message_type",
  "data": { ... }
}
```

#### Binary Messages

Binary messages are sent directly as image data (JPEG/PNG).

### Protocol Flow

1. **Connection:**
   - Client connects to WebSocket
   - Server sends a status message

2. **Settings Update:**
   - Client sends settings as JSON with type="settings"
   - Server applies settings and responds with confirmation

3. **Frame Processing:**
   - Client sends binary frame data
   - Server processes with StreamDiffusion
   - Server sends JSON metadata with type="processed_frame"
   - Server sends binary processed frame

4. **Error Handling:**
   - Server sends error messages as JSON with type="error"

## Troubleshooting

If you encounter issues:

1. Check CUDA with `nvidia-smi`
2. Verify PyTorch CUDA support:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
3. Check logs for detailed error messages
