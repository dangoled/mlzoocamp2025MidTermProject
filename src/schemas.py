from pydantic import BaseModel, Field
from typing import Optional

class StudentData(BaseModel):
    Gender: str = Field(..., description="Student's gender")
    Age: int = Field(..., ge=17, le=30, description="Student's age")
    Course: str = Field(..., description="Student's course/program")
    YearOfStudy: str = Field(..., description="Year of study")
    CGPA: float = Field(..., ge=0, le=4, description="Cumulative GPA")
    StudyStressLevel: int = Field(..., ge=1, le=5, description="Study stress level (1-5)")
    SleepQuality: int = Field(..., ge=1, le=5, description="Sleep quality (1-5)")
    StudyHoursPerWeek: int = Field(..., ge=0, le=40, description="Study hours per week")
    AcademicEngagement: int = Field(..., ge=1, le=5, description="Academic engagement level (1-5)")
    SymptomFrequency_Last7Days: int = Field(..., ge=0, le=7, description="Symptom frequency in last 7 days")

class PredictionResponse(BaseModel):
    mental_health_risk: float = Field(..., ge=0, le=1, description="Probability of mental health issue")
    has_mental_health_issue: bool = Field(..., description="Prediction of mental health issue")
    message: str = Field(..., description="Interpretation message")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API status")