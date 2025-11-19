from fastapi import FastAPI, HTTPException
from src.schemas import StudentData, PredictionResponse, HealthResponse
from src.predict import get_predictor
import uvicorn

app = FastAPI(
    title="University Student Mental Health Prediction API",
    description="API for predicting mental health issues in university students",
    version="1.0.0"
)

# Initialize predictor
predictor = get_predictor()

@app.get("/", response_model=HealthResponse)
async def root():
    return {"status": "University Student Mental Health Prediction API is running"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_mental_health(student_data: StudentData):
    """
    Predict mental health risk for a university student
    
    - **Gender**: Student's gender (Male/Female)
    - **Age**: Student's age (17-30)
    - **Course**: Student's course/program
    - **YearOfStudy**: Year of study (Year 1-Year 4)
    - **CGPA**: Cumulative GPA (0.0-4.0)
    - **StudyStressLevel**: Study stress level (1-5)
    - **SleepQuality**: Sleep quality (1-5)
    - **StudyHoursPerWeek**: Study hours per week (0-40)
    - **AcademicEngagement**: Academic engagement level (1-5)
    - **SymptomFrequency_Last7Days**: Symptom frequency in last 7 days (0-7)
    """
    try:
        prediction = predictor.predict(student_data)
        return PredictionResponse(**prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)