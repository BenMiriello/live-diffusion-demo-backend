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

### 4. Implement the StreamDiffusion Service

Replace the placeholder DiffusionService in `app/services/diffusion.py` with actual StreamDiffusion implementation:

1. Add the following imports at the top of the file:
   ```python
   from streamdiffusion import StreamDiffusion
   from streamdiffusion.schedulers import KarrasDiffusionSchedulers
   from diffusers import AutoencoderKL, UNet2DConditionModel
   from transformers import CLIPTextModel, CLIPTokenizer
   import torch
   ```

2. Update the DiffusionService class to use StreamDiffusion for real-time processing.

3. Make sure the `process_frame` method uses StreamDiffusion to generate images from webcam frames.

### Implementation Example

Here's a basic example of how to implement StreamDiffusion in the `DiffusionService` class:

```python
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
