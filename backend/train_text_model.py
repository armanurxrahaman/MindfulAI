"""
Text Emotion Classifier Training Script
Fine-tune the RoBERTa model for better accuracy on mental health text
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import json
import os

class TextEmotionTrainer:
    def __init__(self, model_name="SamLowe/roberta-base-go_emotions"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Emotion labels
        self.emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral"
        ]
        self.num_labels = len(self.emotion_labels)
        self.label2id = {label: idx for idx, label in enumerate(self.emotion_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
    
    def create_sample_dataset(self):
        """
        Create a sample dataset for demonstration
        In practice, you would load your actual mental health dataset
        """
        # For multi-label, use lists of labels (simulate multi-label)
        sample_data = [
            {"text": "I'm feeling really happy today!", "labels": ["joy"]},
            {"text": "I'm so grateful for my friends", "labels": ["gratitude"]},
            {"text": "I'm feeling anxious about the future", "labels": ["fear"]},
            {"text": "I'm really angry about what happened", "labels": ["anger"]},
            {"text": "I feel sad and lonely", "labels": ["sadness"]},
            {"text": "I'm excited about the new opportunities", "labels": ["excitement"]},
            {"text": "I'm confused about what to do", "labels": ["confusion"]},
            {"text": "I'm proud of my accomplishments", "labels": ["pride"]},
            {"text": "I'm disappointed with the results", "labels": ["disappointment"]},
            {"text": "I'm feeling optimistic about tomorrow", "labels": ["optimism"]},
            {"text": "I'm feeling happy and grateful!", "labels": ["joy", "gratitude"]},
            {"text": "I'm sad and angry", "labels": ["sadness", "anger"]},
            {"text": "I'm nervous and excited", "labels": ["nervousness", "excitement"]},
            {"text": "I'm surprised and happy", "labels": ["surprise", "joy"]},
            {"text": "I'm disappointed and sad", "labels": ["disappointment", "sadness"]},
            {"text": "I'm proud and optimistic", "labels": ["pride", "optimism"]},
            {"text": "I'm grateful and caring", "labels": ["gratitude", "caring"]},
            {"text": "I'm angry and annoyed", "labels": ["anger", "annoyance"]},
            {"text": "I'm happy and excited", "labels": ["joy", "excitement"]},
            {"text": "I'm sad and remorseful", "labels": ["sadness", "remorse"]},
        ]
        # Convert labels to one-hot encoding (as float for BCEWithLogitsLoss)
        for item in sample_data:
            one_hot = [0.0] * self.num_labels
            for label in item["labels"]:
                one_hot[self.label2id[label]] = 1.0
            item["labels"] = one_hot
        return sample_data
    
    def tokenize_function(self, examples):
        """Tokenize the text data"""
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def prepare_dataset(self, data):
        """Prepare dataset for training"""
        # Split data
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)
        
        # Tokenize datasets
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)
        
        return train_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute multi-label accuracy and F1 metrics"""
        logits, labels = eval_pred
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        # Use 0.5 as threshold for multi-label
        preds = (probs > 0.5).astype(int)
        labels = np.array(labels)
        # Subset accuracy: all labels must match
        subset_acc = np.mean(np.all(preds == labels, axis=1))
        # Macro F1
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        # Micro F1
        micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
        return {
            "subset_accuracy": subset_acc,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1
        }
    
    def train(self, output_dir="./improved_text_model"):
        """Train the model"""
        print("🚀 Starting text emotion model training...")
        
        # Create sample dataset (replace with your actual data)
        sample_data = self.create_sample_dataset()
        train_dataset, test_dataset = self.prepare_dataset(sample_data)
        
        print(f"📊 Training samples: {len(train_dataset)}")
        print(f"📊 Test samples: {len(test_dataset)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=5,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to=None,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            warmup_steps=100,
            logging_dir=f"{output_dir}/logs",
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train the model
        print("🏋️ Training model...")
        trainer.train()
        
        # Evaluate the model
        print("📈 Evaluating model...")
        results = trainer.evaluate()
        
        print(f"✅ Training completed!")
        print(f"📊 Subset Accuracy: {results.get('eval_subset_accuracy', 0):.4f}")
        print(f"📊 Macro F1: {results.get('eval_macro_f1', 0):.4f}")
        print(f"📊 Micro F1: {results.get('eval_micro_f1', 0):.4f}")
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"💾 Model saved to: {output_dir}")
        
        return results
    
    def load_trained_model(self, model_path):
        """Load a trained model"""
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        print(f"✅ Loaded trained model from: {model_path}")

def main():
    """Main training function"""
    trainer = TextEmotionTrainer()
    
    # Train the model
    results = trainer.train()
    
    # Save training results
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("🎉 Training completed successfully!")
    print("📁 Check training_results.json for detailed metrics")

if __name__ == "__main__":
    main() 