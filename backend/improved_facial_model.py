"""
Improved Facial Emotion Recognition Model
Using pre-trained ResNet architecture for better accuracy
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
from PIL import Image

class ImprovedEmotionCNN(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(ImprovedEmotionCNN, self).__init__()
        
        # Use pre-trained ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        num_features = self.backbone.fc.in_features
        
        # Add custom classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class ImprovedFacialEmotionRecognizer:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize OpenCV face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize the model
        self.model = ImprovedEmotionCNN(num_classes=7)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded pre-trained model from: {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Emotion labels
        self.emotion_labels = [
            "angry", "disgust", "fear", "happy",
            "sad", "surprise", "neutral"
        ]
        
        # Improved image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet standard size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def detect_face_opencv(self, image):
        """Use OpenCV for face detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Return the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            return [(x, y, w, h)]
        
        return []
    
    def preprocess_face(self, face_img):
        """Improved face preprocessing"""
        # Convert to PIL Image
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        
        # Apply transformations
        face_tensor = self.transform(face_pil)
        return face_tensor.unsqueeze(0).to(self.device)
    
    def predict_emotion(self, image):
        """Predict emotion from facial image with improved accuracy"""
        # Detect face using OpenCV
        faces = self.detect_face_opencv(image)
        
        if len(faces) == 0:
            return None
        
        # Get the first face
        x, y, w, h = faces[0]
        
        # Add padding to face region
        padding = int(min(w, h) * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        face_img = image[y:y+h, x:x+w]
        
        # Preprocess face
        face_tensor = self.preprocess_face(face_img)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            emotion_idx = probabilities.argmax().item()
            confidence = probabilities[0][emotion_idx].item()
        
        return {
            "emotion": self.emotion_labels[emotion_idx],
            "confidence": confidence,
            "face_location": (x, y, w, h),
            "all_probabilities": {
                label: float(prob) 
                for label, prob in zip(self.emotion_labels, probabilities[0])
            }
        }
    
    def get_suggestions(self, emotion: str) -> list:
        """Get personalized suggestions based on detected emotion"""
        suggestions = {
            "happy": [
                "Share your positive mood with others",
                "Document this moment in your journal",
                "Try to maintain this positive energy",
                "Express your gratitude to someone"
            ],
            "sad": [
                "Take a short walk outside",
                "Practice deep breathing exercises",
                "Listen to uplifting music",
                "Reach out to a friend or family member",
                "Write down three things you're grateful for"
            ],
            "angry": [
                "Take deep breaths",
                "Count to 10 slowly",
                "Write down your feelings",
                "Go for a walk to cool down",
                "Try progressive muscle relaxation"
            ],
            "fear": [
                "Practice grounding techniques",
                "Write down your worries",
                "Talk to someone you trust",
                "Try progressive muscle relaxation",
                "Focus on what you can control"
            ],
            "surprise": [
                "Take a moment to process",
                "Share your surprise with someone",
                "Write about what surprised you",
                "Reflect on how this affects you"
            ],
            "disgust": [
                "Identify what triggered this feeling",
                "Take a break from the situation",
                "Practice self-compassion",
                "Focus on something positive"
            ],
            "neutral": [
                "Check in with yourself",
                "Try a new activity",
                "Practice mindfulness",
                "Set a small goal for today",
                "Connect with someone"
            ]
        }
        
        return suggestions.get(emotion, ["Take a moment to reflect on your feelings"])

# Training script for the improved model
def train_improved_facial_model(dataset_path, output_path="./improved_facial_model.pth"):
    """
    Train the improved facial emotion recognition model
    
    Args:
        dataset_path: Path to the emotion dataset
        output_path: Path to save the trained model
    """
    # This would be implemented for training the model
    # For now, we'll use a pre-trained model
    pass

# Example usage
if __name__ == "__main__":
    # Initialize the improved recognizer
    recognizer = ImprovedFacialEmotionRecognizer()
    
    # Example: Load an image and predict emotion
    # image = cv2.imread("test_image.jpg")
    # result = recognizer.predict_emotion(image)
    # print(f"Detected emotion: {result['emotion']} with confidence: {result['confidence']:.2f}")
    
    print("✅ Improved facial emotion recognizer initialized!") 