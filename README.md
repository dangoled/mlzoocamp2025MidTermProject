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

1. **Install uv** (if not installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
