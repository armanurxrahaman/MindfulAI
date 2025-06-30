from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import re
import os

class TextEmotionClassifier:
    def __init__(self, model_path="./improved_text_model"):
        """
        Use the improved fine-tuned GoEmotions model for best accuracy
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            # fallback to original
            self.tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
            self.model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.model.to(self.device)
        self.model.eval()
        
        # Emotion labels from the model
        self.emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral"
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Simple but effective text preprocessing for better accuracy"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
        return text
    
    def predict(self, text: str) -> dict:
        """
        Predict emotion from text input with confidence scores
        """
        # Preprocess
        text = self.preprocess_text(text)
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)[0].cpu().numpy()
        # Get top 3 emotions and their scores
        top_3_indices = np.argsort(probabilities)[::-1][:3]
        top_3_scores = probabilities[top_3_indices]
        emotion_scores = {
            self.emotion_labels[idx]: float(probabilities[idx])
            for idx in top_3_indices
        }
        # Get primary emotion (highest score)
        primary_emotion = self.emotion_labels[top_3_indices[0]]
        confidence = float(top_3_scores[0])
        # Map to basic emotions for suggestions
        basic_emotion = self._map_to_basic_emotion(primary_emotion)
        return {
            "emotion": basic_emotion,
            "confidence": confidence,
            "all_emotions": emotion_scores,
            "detailed_emotion": primary_emotion
        }
    
    def _map_to_basic_emotion(self, detailed_emotion: str) -> str:
        """
        Map detailed emotions to basic emotions for suggestions
        """
        emotion_mapping = {
            # Positive emotions
            "joy": "happy",
            "excitement": "happy",
            "amusement": "happy",
            "gratitude": "happy",
            "love": "happy",
            "optimism": "happy",
            "pride": "happy",
            "approval": "happy",
            "admiration": "happy",
            "caring": "happy",
            
            # Negative emotions
            "anger": "angry",
            "annoyance": "angry",
            "disapproval": "angry",
            
            "fear": "fear",
            "nervousness": "fear",
            
            "sadness": "sad",
            "grief": "sad",
            "disappointment": "sad",
            "remorse": "sad",
            
            "disgust": "disgust",
            
            "surprise": "surprise",
            
            # Neutral
            "neutral": "neutral",
            "confusion": "neutral",
            "curiosity": "neutral",
            "realization": "neutral"
        }
        
        return emotion_mapping.get(detailed_emotion, "neutral")
    
    def get_suggestions(self, emotion: str) -> list:
        """
        Get personalized suggestions based on detected emotion
        """
        suggestions = {
            "happy": [
                "Share your positive mood with friends",
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