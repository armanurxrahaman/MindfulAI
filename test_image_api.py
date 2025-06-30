import requests

# Path to a test image (replace with your own image path)
image_path = "test_face.jpg"  # Make sure this file exists in your project root

url = "http://localhost:8000/analyze/image"

with open(image_path, "rb") as f:
    files = {"file": (image_path, f, "image/jpeg")}
    response = requests.post(url, files=files)

print("Status code:", response.status_code)
try:
    print("Response:", response.json())
except Exception:
    print("Raw response:", response.text) 