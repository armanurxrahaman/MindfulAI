"""
Model Evaluation Script
Evaluate the accuracy of text and facial emotion recognition models
"""

import numpy as np
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report

class ModelEvaluator:
    def __init__(self):
        self.results = {}
    
    def evaluate_text_model(self, test_data, predictions, model_name="text_emotion_classifier"):
        """Evaluate text emotion classification model"""
        print(f"📊 Evaluating {model_name}...")
        
        # Extract true labels
        true_labels = [item['label'] for item in test_data]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        
        # Store results
        self.results[model_name] = {
            "accuracy": accuracy,
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"],
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ {model_name} Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {report['weighted avg']['precision']:.4f}")
        print(f"   Recall: {report['weighted avg']['recall']:.4f}")
        print(f"   F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        return self.results[model_name]
    
    def generate_report(self, output_file="accuracy_report.json"):
        """Generate accuracy report"""
        report = {
            "evaluation_date": datetime.now().isoformat(),
            "models": self.results,
            "summary": {
                "total_models": len(self.results),
                "average_accuracy": np.mean([result["accuracy"] for result in self.results.values()]),
                "best_model": max(self.results.items(), key=lambda x: x[1]["accuracy"])[0] if self.results else None,
                "best_accuracy": max([result["accuracy"] for result in self.results.values()]) if self.results else 0
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Report saved to: {output_file}")
        return report

def main():
    evaluator = ModelEvaluator()
    
    # Sample test data
    test_data = [
        {"text": "I'm happy!", "label": "joy"},
        {"text": "I'm sad", "label": "sadness"},
        {"text": "I'm angry", "label": "anger"},
    ]
    
    predictions = ["joy", "sadness", "anger"]
    
    # Evaluate
    evaluator.evaluate_text_model(test_data, predictions)
    evaluator.generate_report()
    
    print("🎉 Evaluation completed!")

if __name__ == "__main__":
    main() 