import cv2
import numpy as np
import requests
import json
import base64
import os
from typing import Optional, Dict, Any
import mediapipe as mp

class CloudFacialRecognizer:
    """Facial emotion recognition using cloud APIs for maximum reliability"""
    
    def __init__(self, api_type="azure"):
        self.api_type = api_type
        self.api_key = os.getenv(f"{api_type.upper()}_API_KEY")
        self.endpoint = os.getenv(f"{api_type.upper()}_ENDPOINT")
        
        # Initialize MediaPipe for face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        
        # Emotion labels
        self.emotion_labels = [
            "angry", "disgust", "fear", "happy",
            "sad", "surprise", "neutral"
        ]
        
        print(f"✅ Cloud facial emotion recognition initialized ({api_type})")
    
    def detect_face_mediapipe(self, image):
        """Use MediaPipe for face detection"""
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
    
    def predict_emotion(self, image) -> Optional[Dict[str, Any]]:
        """Predict emotion using cloud API"""
        if not self.api_key:
            print("⚠️ No API key found, using fallback method")
            return self._fallback_analysis(image)
        
        # Detect face
        faces = self.detect_face_mediapipe(image)
        if not faces:
            return None
        
        x, y, w, h = faces[0]
        
        # Crop face with padding
        padding = int(min(w, h) * 0.2)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        face_img = image[y:y+h, x:x+w]
        
        if face_img.shape[0] < 64 or face_img.shape[1] < 64:
            return None
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', face_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        try:
            if self.api_type == "azure":
                return self._azure_emotion_analysis(img_base64, (x, y, w, h))
            elif self.api_type == "google":
                return self._google_emotion_analysis(img_base64, (x, y, w, h))
            else:
                return self._fallback_analysis(image)
        except Exception as e:
            print(f"⚠️ Cloud API failed: {e}")
            return self._fallback_analysis(image)
    
    def _azure_emotion_analysis(self, img_base64, face_location):
        """Analyze emotion using Azure Face API"""
        headers = {
            'Content-Type': 'application/octet-stream',
            'Ocp-Apim-Subscription-Key': self.api_key
        }
        
        # Decode base64 image
        img_data = base64.b64decode(img_base64)
        
        # Azure Face API endpoint
        url = f"{self.endpoint}/face/v1.0/detect?returnFaceAttributes=emotion"
        
        response = requests.post(url, headers=headers, data=img_data)
        
        if response.status_code == 200:
            faces = response.json()
            if faces:
                face = faces[0]
                emotions = face['faceAttributes']['emotion']
                
                # Map Azure emotions to our labels
                emotion_mapping = {
                    'anger': 'angry',
                    'contempt': 'disgust',
                    'disgust': 'disgust',
                    'fear': 'fear',
                    'happiness': 'happy',
                    'sadness': 'sad',
                    'surprise': 'surprise',
                    'neutral': 'neutral'
                }
                
                # Get the emotion with highest confidence
                max_emotion = max(emotions.items(), key=lambda x: x[1])
                emotion = emotion_mapping.get(max_emotion[0], 'neutral')
                confidence = max_emotion[1]
                
                # Create probabilities dict
                probabilities = {}
                for azure_emotion, confidence_val in emotions.items():
                    mapped_emotion = emotion_mapping.get(azure_emotion, 'neutral')
                    if mapped_emotion in probabilities:
                        probabilities[mapped_emotion] = max(probabilities[mapped_emotion], confidence_val)
                    else:
                        probabilities[mapped_emotion] = confidence_val
                
                # Create top predictions
                sorted_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                top_predictions = [
                    {"emotion": emotion, "confidence": float(conf)}
                    for emotion, conf in sorted_emotions[:3]
                ]
                
                return {
                    "emotion": emotion,
                    "confidence": confidence,
                    "face_location": face_location,
                    "all_probabilities": probabilities,
                    "top_predictions": top_predictions,
                    "api_used": "azure"
                }
        
        return None
    
    def _google_emotion_analysis(self, img_base64, face_location):
        """Analyze emotion using Google Cloud Vision API"""
        headers = {
            'Content-Type': 'application/json'
        }
        
        # Google Cloud Vision API request
        request_data = {
            "requests": [
                {
                    "image": {
                        "content": img_base64
                    },
                    "features": [
                        {
                            "type": "FACE_DETECTION",
                            "maxResults": 1
                        }
                    ]
                }
            ]
        }
        
        url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"
        
        response = requests.post(url, headers=headers, json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            if 'responses' in result and result['responses']:
                face_annotations = result['responses'][0].get('faceAnnotations', [])
                if face_annotations:
                    face = face_annotations[0]
                    
                    # Initialize probabilities
                    probabilities = {
                        'happy': 0.1,
                        'sad': 0.1,
                        'angry': 0.1,
                        'fear': 0.1,
                        'surprise': 0.1,
                        'disgust': 0.1,
                        'neutral': 0.4
                    }
                    
                    # Analyze joy likelihood
                    joy_likelihood = face.get('joyLikelihood', 'UNLIKELY')
                    if joy_likelihood in ['LIKELY', 'VERY_LIKELY']:
                        probabilities['happy'] = 0.8 if joy_likelihood == 'LIKELY' else 0.9
                        probabilities['neutral'] = 0.1
                    
                    # Analyze sorrow likelihood
                    sorrow_likelihood = face.get('sorrowLikelihood', 'UNLIKELY')
                    if sorrow_likelihood in ['LIKELY', 'VERY_LIKELY']:
                        probabilities['sad'] = 0.8 if sorrow_likelihood == 'LIKELY' else 0.9
                        probabilities['neutral'] = 0.1
                    
                    # Analyze anger likelihood
                    anger_likelihood = face.get('angerLikelihood', 'UNLIKELY')
                    if anger_likelihood in ['LIKELY', 'VERY_LIKELY']:
                        probabilities['angry'] = 0.8 if anger_likelihood == 'LIKELY' else 0.9
                        probabilities['neutral'] = 0.1
                    
                    # Analyze surprise likelihood
                    surprise_likelihood = face.get('surpriseLikelihood', 'UNLIKELY')
                    if surprise_likelihood in ['LIKELY', 'VERY_LIKELY']:
                        probabilities['surprise'] = 0.8 if surprise_likelihood == 'LIKELY' else 0.9
                        probabilities['neutral'] = 0.1
                    
                    # Analyze additional facial features
                    if 'boundingPoly' in face:
                        # Check for wide eyes (fear/surprise)
                        vertices = face['boundingPoly']['vertices']
                        if len(vertices) >= 4:
                            width = abs(vertices[1]['x'] - vertices[0]['x'])
                            height = abs(vertices[2]['y'] - vertices[1]['y'])
                            aspect_ratio = width / height if height > 0 else 1
                            
                            if aspect_ratio > 1.2:  # Wide face (surprise/fear)
                                probabilities['surprise'] = max(probabilities['surprise'], 0.6)
                                probabilities['fear'] = max(probabilities['fear'], 0.5)
                    
                    # Get the emotion with highest probability
                    emotion = max(probabilities, key=probabilities.get)
                    confidence = probabilities[emotion]
                    
                    # Normalize probabilities
                    total = sum(probabilities.values())
                    if total > 0:
                        probabilities = {k: v/total for k, v in probabilities.items()}
                    
                    # Create top predictions
                    sorted_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                    top_predictions = [
                        {"emotion": emotion, "confidence": float(conf)}
                        for emotion, conf in sorted_emotions[:3]
                    ]
                    
                    return {
                        "emotion": emotion,
                        "confidence": confidence,
                        "face_location": face_location,
                        "all_probabilities": probabilities,
                        "top_predictions": top_predictions,
                        "api_used": "google"
                    }
        
        return None
    
    def _fallback_analysis(self, image):
        """Fallback analysis using OpenCV"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Initialize probabilities
        probabilities = {emotion: 0.0 for emotion in self.emotion_labels}
        
        try:
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                roi_gray = gray[y:y+h, x:x+w]
                
                # Detect smile
                smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
                
                if len(smiles) > 0:
                    probabilities["happy"] = 0.8
                    probabilities["neutral"] = 0.1
                else:
                    probabilities["neutral"] = 0.7
                    probabilities["sad"] = 0.2
                
                # Normalize
                total = sum(probabilities.values())
                if total > 0:
                    probabilities = {k: v/total for k, v in probabilities.items()}
            
        except Exception as e:
            probabilities["neutral"] = 1.0
        
        # Get most likely emotion
        emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[emotion]
        
        top_predictions = [
            {"emotion": emotion, "confidence": float(confidence)}
        ]
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "face_location": (0, 0, image.shape[1], image.shape[0]),
            "all_probabilities": probabilities,
            "top_predictions": top_predictions,
            "api_used": "fallback"
        }
    
    def get_suggestions(self, emotion: str) -> list:
        """Get personalized suggestions based on detected emotion"""
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
                "Connect with someone you haven't talked to recently"
            ]
        }
        
        return suggestions.get(emotion, ["Take a moment to reflect on your feelings"]) 
