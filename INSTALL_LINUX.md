# Linux Installation Guide for StreamDiffusion Backend

This guide walks through setting up the Live Diffusion Demo backend specifically for Debian Linux with an RTX 3090 GPU.

## CUDA and PyTorch Requirements

The application requires:
- NVIDIA GPU (RTX 3090)
- CUDA 11.8
- cuDNN
- PyTorch with CUDA support

## Step-by-Step Installation

### 1. Verify GPU and CUDA Setup

First, check if your NVIDIA drivers are correctly installed:

```bash
nvidia-smi
```

You should see information about your RTX 3090 and CUDA version. If not, install the NVIDIA drivers and CUDA toolkit.

### 2. Create and Activate Virtual Environment

```bash
# Navigate to the backend directory
cd backend

# Create a virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA 11.8

```bash
# Install PyTorch with CUDA 11.8 support (do not use pip install -r requirements.txt yet)
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### 4. Install StreamDiffusion and Other Dependencies

```bash
# Install StreamDiffusion - a specific version compatible with CUDA 11.8
pip install streamdiffusion==0.7.1

# Install other dependencies
pip install -r requirements.txt --ignore-installed torch torchvision
```

### 5. Create .env File

Ensure your .env file is set up correctly:

```bash
cp .env.example .env
```

Edit the `.env` file with the following settings:

```
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info

# CORS Settings
CORS_ORIGINS=http://localhost:5173,http://localhost:4173

# StreamDiffusion Settings
MODEL_ID=stabilityai/sd-turbo
GPU_DEVICE=0
DENOISING_STRENGTH=0.5
```

### 6. Verify PyTorch CUDA Support

Run this Python script to verify CUDA is properly configured with PyTorch:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'NA'); print('Device count:', torch.cuda.device_count()); print('Current device:', torch.cuda.current_device()); print('Device name:', torch.cuda.get_device_name(0))"
```

You should see output confirming CUDA is available and recognizing your RTX 3090.

### 7. Run the Backend

```bash
python run.py
```

## Troubleshooting

### Common Issues

1. **CUDA not found**: If PyTorch doesn't detect CUDA, ensure your NVIDIA drivers and CUDA are correctly installed. Use `nvidia-smi` to check.

2. **StreamDiffusion import errors**: If you see import errors related to StreamDiffusion, ensure you've installed the compatible versions of PyTorch and its dependencies.

3. **Memory errors**: If you encounter CUDA out of memory errors, try:
   - Reducing the model width/height (currently set to 512x512)
   - Adjusting batch sizes
   - Closing other GPU-intensive applications

4. **Environment Variables**: If you see errors about CORS_ORIGINS or other config values, double-check your .env file format.

### Advanced: Using a Different CUDA Version

If you need to use a different CUDA version, adjust the PyTorch installation accordingly:

- For CUDA 11.7: `pip install torch==2.1.2+cu117 torchvision==0.16.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117`
- For CUDA 12.1: `pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121`
