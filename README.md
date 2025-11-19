# University Student Mental Health Prediction

A machine learning project to predict mental health issues in university students based on academic and demographic factors.


## Problem Description

This binary classification Midterm project aims at predicting mental health conditions (Depression, Anxiety, Panic Attacks) in university students based on their academic and demographic characteristics. The model can help identify at-risk students for early intervention and support services.

Relevance: This is a highly relevant real-world problem as early detection of depression risk in students using simple survey questions can help universities provide timely mental health support. Mental health among students is real and early diagnosis can save alot of harm.

The project is built using:

- uv for virtual environment and dependency management
- FastAPI + Pydantic
- Pickle for model saving
- Docker for containerization

This project  is done using the Kaggle *Student Mental Health* dataset (downloaded from https://www.kaggle.com/datasets/shariful07/student-mental-health). 



## Project Structure

- `notebooks/`: Jupyter notebooks for EDA
- `src/`: Source code for training and prediction
- `app/`: FastAPI application
- `models/`: Trained models and artifacts
- `data/`: Dataset
- `scripts/`: Utility scripts

## Setup

1. Install uv
2. Intialize the project:
      uv init
3. Create virtual environment and install dependencies:
      uv add scikit-learn fastapi uvicorn
4. Add development dependency
      uv add --dev requests

## Usage
1. Train the model
      python scripts/train_model.py
2. Run the API locally
      uvicorn app.main:app --reload
3. Test the API
      python scripts/test_api.py

### API Endpoints
  - GET /: Health check
  - GET /health: API status
  - POST /predict: Predict mental health risk

## Deployment
1. Build and run with Docker
    docker build -t mental-health-api .
    docker run -p 8000:8000 mental-health-api
2. Deploy to Fly.io
    flyctl auth login
    flyctl deploy
3. Test with curl
curl -X POST "http://localhost:8000/predict" \ 

     -H "Content-Type: application/json" \ 
     
     -d '{ 
     
       "Gender": "Female",
   
       "Age": 21,
       "Course": "Engineering", 
       "YearOfStudy": "Year 2",
       "CGPA": 3.2,
       "StudyStressLevel": 4,
       "SleepQuality": 2,
       "StudyHoursPerWeek": 25,
       "AcademicEngagement": 3,
       "SymptomFrequency_Last7Days": 5
     }'
