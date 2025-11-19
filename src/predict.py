import pickle
import json
import pandas as pd
from pathlib import Path

class MentalHealthPredictor:
    def __init__(self, model_path: str = "models/model.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and feature names"""
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            
            # Load feature names
            with open("models/feature_names.json", "r") as f:
                self.feature_names = json.load(f)
                
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, student_data):
        """Make prediction for student data"""
        # Convert to DataFrame
        input_df = pd.DataFrame([student_data.dict()])
        
        # Make prediction
        probability = self.model.predict_proba(input_df)[0, 1]
        prediction = probability > 0.5
        
        # Create interpretation message
        if probability > 0.7:
            message = "High risk of mental health issues - recommend support services"
        elif probability > 0.4:
            message = "Moderate risk - monitor and provide resources"
        else:
            message = "Low risk - continue regular check-ins"
        
        return {
            "mental_health_risk": round(probability, 4),
            "has_mental_health_issue": bool(prediction),
            "message": message
        }

# Global predictor instance
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = MentalHealthPredictor()
    return predictor