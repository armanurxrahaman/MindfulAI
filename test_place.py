import cv2
import os
from PIL import Image

# Check current working directory
print(f"Current working directory: {os.getcwd()}")
print(f"test_face.jpg exists: {os.path.exists('test_face.jpg')}")

# Try to read the image with OpenCV
img = cv2.imread("test_face.jpg")
print(f"Image read successfully (OpenCV): {img is not None}")

if img is not None:
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
else:
    # Try with absolute path
    abs_path = os.path.abspath("test_face.jpg")
    print(f"Absolute path: {abs_path}")
    img = cv2.imread(abs_path)
    print(f"Image read with absolute path (OpenCV): {img is not None}")
    
    if img is not None:
        print(f"Image shape: {img.shape}")
        print(f"Image dtype: {img.dtype}")
    else:
        # Try with forward slashes
        forward_path = abs_path.replace("\\", "/")
        print(f"Forward slash path: {forward_path}")
        img = cv2.imread(forward_path)
        print(f"Image read with forward slashes (OpenCV): {img is not None}")
        
        if img is not None:
            print(f"Image shape: {img.shape}")
            print(f"Image dtype: {img.dtype}")

# Try to open with PIL
try:
    pil_img = Image.open("test_face.jpg")
    pil_img.verify()  # Verify image integrity
    print("Image opened successfully with PIL.")
except Exception as e:
    print(f"PIL failed to open image: {e}")