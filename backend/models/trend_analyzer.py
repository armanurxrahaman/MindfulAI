import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import pandas as pd

class TrendAnalyzer:
    def __init__(self):
        self.model = LinearRegression()
        self.emotion_scores = {
            "joy": 1.0,
            "happy": 1.0,
            "neutral": 0.5,
            "surprise": 0.5,
            "sad": 0.0,
            "sadness": 0.0,
            "fear": -0.5,
            "anger": -0.5,
            "angry": -0.5,
            "disgust": -1.0
        }
    
    def _convert_emotion_to_score(self, emotion: str) -> float:
        """
        Convert emotion to numerical score
        """
        return self.emotion_scores.get(emotion.lower(), 0.0)
    
    def _prepare_data(self, dates: list, emotions: list) -> tuple:
        """
        Prepare data for trend analysis
        """
        # Convert dates to numerical values (days since first entry)
        first_date = datetime.strptime(dates[0], "%Y-%m-%d")
        X = np.array([
            (datetime.strptime(date, "%Y-%m-%d") - first_date).days
            for date in dates
        ]).reshape(-1, 1)
        
        # Convert emotions to scores
        y = np.array([self._convert_emotion_to_score(emotion) for emotion in emotions])
        
        return X, y
    
    def analyze_trend(self, dates: list, emotions: list) -> dict:
        """
        Analyze emotional trend and predict future mood
        """
        if len(dates) < 2:
            return {
                "trend": "insufficient_data",
                "prediction": None,
                "confidence": 0.0
            }
        
        # Prepare data
        X, y = self._prepare_data(dates, emotions)
        
        # Fit model
        self.model.fit(X, y)
        
        # Calculate trend
        slope = self.model.coef_[0]
        if slope > 0.1:
            trend = "improving"
        elif slope < -0.1:
            trend = "declining"
        else:
            trend = "stable"
        
        # Predict next day's mood
        last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
        next_date = last_date + timedelta(days=1)
        next_day = (next_date - datetime.strptime(dates[0], "%Y-%m-%d")).days
        
        prediction = self.model.predict([[next_day]])[0]
        
        # Calculate confidence based on R² score
        confidence = self.model.score(X, y)
        
        return {
            "trend": trend,
            "prediction": float(prediction),
            "confidence": float(confidence)
        }
    
    def get_insights(self, trend: str, prediction: float) -> list:
        """
        Generate insights based on trend analysis
        """
        insights = []
        
        if trend == "improving":
            insights.extend([
                "Your mood has been improving! Keep up the positive momentum.",
                "Consider what activities or thoughts have contributed to this improvement.",
                "Try to maintain these positive habits."
            ])
        elif trend == "declining":
            insights.extend([
                "I notice your mood has been declining. Remember, it's okay to not be okay.",
                "Consider reaching out to friends, family, or a mental health professional.",
                "Try to identify any patterns or triggers that might be affecting your mood."
            ])
        else:
            insights.extend([
                "Your mood has been relatively stable.",
                "This could be a good time to try new activities or set new goals.",
                "Consider what brings you joy and try to incorporate more of that into your routine."
            ])
        
        # Add prediction-based insights
        if prediction > 0.7:
            insights.append("The trend suggests you'll be feeling positive tomorrow!")
        elif prediction < 0.3:
            insights.append("The trend suggests you might feel down tomorrow. Remember to practice self-care.")
        
        return insights 