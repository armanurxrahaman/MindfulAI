import cv2
from deepface import DeepFace

class FacialEmotionRecognizer:
    def __init__(self):
        self.emotion_labels = [
            "angry", "disgust", "fear", "happy",
            "sad", "surprise", "neutral"
        ]
        print("✅ DeepFace facial emotion recognition initialized (offline, no cloud)")

    def detect_face_mediapipe(self, image):
        # Not needed for DeepFace, but kept for API compatibility
        h, w, _ = image.shape
        return [(0, 0, w, h)]

    def predict_emotion(self, image):
        # DeepFace expects BGR (OpenCV) or RGB (numpy), both work
        try:
            result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
            print("DeepFace result:", result)  # Debug print
            
            # DeepFace returns a list, get the first result
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            emotion = result['dominant_emotion']
            confidence = result['emotion'][emotion] / 100.0 if result['emotion'][emotion] > 1 else result['emotion'][emotion]
            all_probabilities = {k: float(v)/100.0 if v > 1 else float(v) for k, v in result['emotion'].items()}
            top_predictions = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
            return {
                "emotion": emotion,
                "confidence": confidence,
                "face_location": (0, 0, image.shape[1], image.shape[0]),
                "all_probabilities": all_probabilities,
                "top_predictions": [
                    {"emotion": emo, "confidence": float(conf)} for emo, conf in top_predictions
                ]
            }
        except Exception as e:
            print(f"DeepFace error: {e}")
            return None

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
                "Connect with someone you haven't talked to recently"
            ]
        }
        return suggestions.get(emotion, ["Take a moment to reflect on your feelings"]) 