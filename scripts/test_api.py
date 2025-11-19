#!/usr/bin/env python3
"""
Script to test the API locally
"""
import requests
import json

# Test data
test_student = {
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
}

def test_api():
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_student
        )
        
        if response.status_code == 200:
            print("✅ API test successful!")
            print("Response:", json.dumps(response.json(), indent=2))
        else:
            print(f"❌ API test failed with status {response.status_code}")
            print("Response:", response.text)
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    test_api()