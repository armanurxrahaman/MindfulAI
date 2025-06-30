from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import cv2
import numpy as np
from datetime import datetime
import json
import os
import sys
from deepface import DeepFace

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our models
from models.text_emotion_classifier import TextEmotionClassifier
from improved_facial_model import ImprovedFacialEmotionRecognizer
from models.trend_analyzer import TrendAnalyzer
from models.content_retriever import ContentRetriever

app = FastAPI(title="Mental Health Support Platform API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
text_classifier = TextEmotionClassifier()
# Use the improved model without requiring the weights file for deployment
facial_recognizer = ImprovedFacialEmotionRecognizer(model_path=None)
trend_analyzer = TrendAnalyzer()
content_retriever = ContentRetriever()

# Data models
class TextInput(BaseModel):
    text: str

class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    suggestions: List[str]
    content: List[Dict]

class TrendData(BaseModel):
    dates: List[str]
    emotions: List[str]
    scores: List[float]
    trend: str
    prediction: Optional[float]
    confidence: float
    insights: List[str]

class UserEntry(BaseModel):
    date: str
    emotion: str
    text: Optional[str]
    image_path: Optional[str]

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to Mental Health Support Platform API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "MindfulAI API is running"}

@app.post("/analyze/text", response_model=EmotionResponse)
async def analyze_text(input_data: TextInput):
    # Get emotion prediction
    emotion_result = text_classifier.predict(input_data.text)
    
    # Get suggestions
    suggestions = text_classifier.get_suggestions(emotion_result["emotion"])
    
    # Get relevant content
    content = content_retriever.retrieve_content(
        query=input_data.text,
        emotion=emotion_result["emotion"]
    )
    
    return {
        "emotion": emotion_result["emotion"],
        "confidence": emotion_result["confidence"],
        "suggestions": suggestions,
        "content": content
    }

@app.post("/analyze/image", response_model=EmotionResponse)
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    try:
        result = DeepFace.analyze(image, actions=['emotion'])
        if isinstance(result, list):
            result = result[0]
        emotion = result['dominant_emotion']
        confidence = result['emotion'][emotion] / 100.0
        suggestions = facial_recognizer.get_suggestions(emotion)
        content = []  # You can use your existing logic here
        return {
            "emotion": emotion,
            "confidence": confidence,
            "suggestions": suggestions,
            "content": content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trends", response_model=TrendData)
async def get_trends():
    # Return empty data until user provides input
    return {
        "dates": [],
        "emotions": [],
        "scores": [],
        "trend": "insufficient_data",
        "prediction": None,
        "confidence": 0.0,
        "insights": []
    }

@app.post("/add_entry")
async def add_entry(entry: UserEntry):
    # In a real application, this would save to a database
    # For now, we'll just return success
    return {"message": "Entry added successfully"}

@app.get("/daily_content")
async def get_daily_content():
    # Get random content for daily inspiration
    content = content_retriever.get_random_content(n_results=3)
    return {"content": content}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 