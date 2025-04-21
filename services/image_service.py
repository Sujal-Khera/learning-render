# services/image_service.py
import cv2
import numpy as np
from PIL import Image

def process_image(image_path, target_size=(224, 224)):
    """Process an image for classification"""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Resize to target size
        img = img.resize(target_size)
        
        # Convert to array
        img_array = np.array(img)
        
        # Handle grayscale images
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Handle RGBA images
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # Normalize pixel values
        img_array = img_array.astype('float32') / 255.0
        
        # Expand dimensions for model input
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
