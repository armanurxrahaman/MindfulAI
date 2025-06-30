import chromadb
from chromadb.utils import embedding_functions
import json
import os
from typing import List, Dict
import numpy as np

class ContentRetriever:
    def __init__(self, persist_directory: str = "data/chroma_db"):
        """
        Initialize the content retriever with ChromaDB
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use sentence-transformers for embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="mental_health_content",
            embedding_function=self.embedding_function
        )
        
        # Commented out: Do not initialize with default content
        # if self.collection.count() == 0:
        #     self._initialize_default_content()
    
    def _initialize_default_content(self):
        """
        Initialize the database with some default mental health content
        """
        default_content = [
            {
                "text": "Practice deep breathing exercises for 5 minutes when feeling stressed. Inhale for 4 counts, hold for 4, exhale for 4.",
                "metadata": {"type": "exercise", "emotion": "stress"}
            },
            {
                "text": "Take a mindful walk in nature. Notice the colors, sounds, and sensations around you.",
                "metadata": {"type": "activity", "emotion": "anxiety"}
            },
            {
                "text": "Write down three things you're grateful for each day. This can help shift focus to positive aspects of life.",
                "metadata": {"type": "practice", "emotion": "depression"}
            },
            {
                "text": "Connect with a friend or family member. Social support is crucial for mental well-being.",
                "metadata": {"type": "social", "emotion": "loneliness"}
            },
            {
                "text": "Try progressive muscle relaxation: tense and release each muscle group for 5 seconds.",
                "metadata": {"type": "exercise", "emotion": "tension"}
            }
        ]
        
        # Add content to collection
        self.collection.add(
            documents=[item["text"] for item in default_content],
            metadatas=[item["metadata"] for item in default_content],
            ids=[f"content_{i}" for i in range(len(default_content))]
        )
    
    def add_content(self, text: str, metadata: Dict):
        """
        Add new content to the database
        """
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[f"content_{self.collection.count()}"]
        )
    
    def retrieve_content(self, query: str, emotion: str = None, n_results: int = 3) -> List[Dict]:
        """
        Retrieve relevant content based on query and emotion
        """
        # Prepare where clause if emotion is specified
        where = {"emotion": emotion} if emotion else None
        
        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": float(results["distances"][0][i])
            })
        
        return formatted_results
    
    def get_emotion_specific_content(self, emotion: str, n_results: int = 3) -> List[Dict]:
        """
        Get content specific to an emotion
        """
        results = self.collection.query(
            query_texts=[""],
            n_results=n_results,
            where={"emotion": emotion}
        )
        
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        
        return formatted_results
    
    def get_random_content(self, n_results: int = 1) -> List[Dict]:
        """
        Get random content from the database
        """
        # Get all content
        all_content = self.collection.get()
        
        # Randomly select n_results items
        if len(all_content["ids"]) > 0:
            indices = np.random.choice(
                len(all_content["ids"]),
                size=min(n_results, len(all_content["ids"])),
                replace=False
            )
            
            return [{
                "text": all_content["documents"][i],
                "metadata": all_content["metadatas"][i]
            } for i in indices]
        
        return [] 