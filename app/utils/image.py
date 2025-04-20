import io
import base64
import logging
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


def decode_image(image_data: bytes) -> Tuple[np.ndarray, int, int]:
    """
    Decode binary image data into a numpy array
    
    Args:
        image_data: Binary image data
        
    Returns:
        Tuple of (image as numpy array, width, height)
    """
    try:
        # Decode the image with PIL
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array (RGB)
        np_image = np.array(image)
        
        # Get dimensions
        height, width = np_image.shape[:2]
        
        return np_image, width, height
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise ValueError(f"Invalid image data: {e}")


def encode_image(
    image: np.ndarray, format: str = "jpeg", quality: int = 90
) -> Tuple[bytes, int, int]:
    """
    Encode a numpy array as a compressed image
    
    Args:
        image: Numpy array containing the image
        format: Output format (jpeg, png)
        quality: Compression quality (0-100, jpeg only)
        
    Returns:
        Tuple of (binary image data, width, height)
    """
    try:
        # Get dimensions
        height, width = image.shape[:2]
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Compress to the desired format
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format.upper(), quality=quality)
        buffer.seek(0)
        
        return buffer.getvalue(), width, height
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        raise ValueError(f"Invalid image data: {e}")


def resize_image(
    image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None
) -> np.ndarray:
    """
    Resize an image to the specified dimensions
    
    Args:
        image: Numpy array containing the image
        width: Target width (if None, will maintain aspect ratio)
        height: Target height (if None, will maintain aspect ratio)
        
    Returns:
        Resized image as numpy array
    """
    try:
        # Get current dimensions
        curr_height, curr_width = image.shape[:2]
        
        # If both dimensions are None, return original
        if width is None and height is None:
            return image
        
        # If one dimension is None, calculate the other to maintain aspect ratio
        if width is None:
            # Calculate width to maintain aspect ratio
            width = int(curr_width * (height / curr_height))
        elif height is None:
            # Calculate height to maintain aspect ratio
            height = int(curr_height * (width / curr_width))
        
        # Resize the image
        resized = cv2.resize(
            image, (width, height), interpolation=cv2.INTER_AREA
        )
        
        return resized
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        raise ValueError(f"Invalid image data: {e}")
