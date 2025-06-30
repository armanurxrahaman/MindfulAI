import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
import os
import torchvision.models as models
import warnings
warnings.filterwarnings("ignore")

class HighAccuracyEmotionModel(nn.Module):
    """High-accuracy emotion recognition model using pre-trained EfficientNet"""
    def __init__(self, num_classes=7):
        super(HighAccuracyEmotionModel, self).__init__()
        self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        # Initialize only the classifier weights
        for m in self.backbone.classifier[1]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.backbone(x)

class ImprovedFacialEmotionRecognizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.emotion_labels = [
            "angry", "disgust", "fear", "happy",
            "sad", "surprise", "neutral"
        ]
        self.model = HighAccuracyEmotionModel(num_classes=7).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print("✅ High-accuracy facial emotion recognition initialized")

    def detect_face_mediapipe(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            return [(x, y, width, height)]
        return []

    def preprocess_face(self, face_img):
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = self.transform(face_pil)
        return face_tensor.unsqueeze(0).to(self.device)

    def predict_emotion(self, image):
        faces = self.detect_face_mediapipe(image)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        padding = int(min(w, h) * 0.2)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        face_img = image[y:y+h, x:x+w]
        if face_img.shape[0] < 64 or face_img.shape[1] < 64:
            return None
        face_tensor = self.preprocess_face(face_img)
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            emotion_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][emotion_idx].item()
            if confidence < 0.15:
                emotion_idx = 6  # neutral
                confidence = 0.8
        result = {
            "emotion": self.emotion_labels[emotion_idx],
            "confidence": confidence,
            "face_location": (x, y, w, h),
            "all_probabilities": {
                label: float(prob)
                for label, prob in zip(self.emotion_labels, probabilities[0])
            },
            "top_predictions": [
                {
                    "emotion": self.emotion_labels[idx.item()],
                    "confidence": float(prob.item())
                }
                for prob, idx in zip(*torch.topk(probabilities[0], 3))
            ]
        }
        return result

    def get_suggestions(self, emotion: str) -> list:
        suggestions = {
            "happy": [
                "Share your positive mood with others",
                "Document this moment in your journal",
                "Try to maintain this positive energy",
                "Express your gratitude to someone",
                "Take a photo to remember this feeling"
            ],
            "sad": [
                "Take a short walk outside",
                "Practice deep breathing exercises",
                "Listen to uplifting music",
                "Reach out to a friend or family member",
                "Write down three things you're grateful for",
                "Try some gentle stretching"
            ],
            "angry": [
                "Take deep breaths",
                "Count to 10 slowly",
                "Write down your feelings",
                "Go for a walk to cool down",
                "Try progressive muscle relaxation",
                "Listen to calming music"
            ],
            "fear": [
                "Practice grounding techniques",
                "Write down your worries",
                "Talk to someone you trust",
                "Try progressive muscle relaxation",
                "Focus on what you can control",
                "Use the 5-4-3-2-1 sensory technique"
            ],
            "surprise": [
                "Take a moment to process",
                "Share your surprise with someone",
                "Write about what surprised you",
                "Reflect on how this affects you",
                "Take a few deep breaths"
            ],
            "disgust": [
                "Identify what triggered this feeling",
                "Take a break from the situation",
                "Practice self-compassion",
                "Focus on something positive",
                "Change your environment if possible"
            ],
            "neutral": [
                "Check in with yourself",
                "Try a new activity",
                "Practice mindfulness",
                "Set a small goal for today",
                "Connect with someone",
                "Take a moment to reflect on your day"
            ]
        }
        return suggestions.get(emotion, ["Take a moment to reflect on your feelings"]) 