import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import json
from pathlib import Path

def prepare_data():
    """Load and prepare the data"""
    df = pd.read_csv('data/mentalhealth_dataset.csv')
    
    # Data cleaning
    df['YearOfStudy'] = df['YearOfStudy'].str.replace('year', 'Year', case=False).str.strip()
    
    # Create target variable
    df['MentalHealthIssue'] = (df['Depression'] | df['Anxiety'] | df['PanicAttack']).astype(int)
    
    # Features and target
    X = df.drop(['MentalHealthIssue', 'Depression', 'Anxiety', 'PanicAttack', 'Timestamp'], axis=1, errors='ignore')
    y = df['MentalHealthIssue']
    
    return X, y

def create_preprocessor():
    """Create preprocessing pipeline"""
    categorical_features = ['Gender', 'Course', 'YearOfStudy']
    numeric_features = ['Age', 'CGPA', 'StudyStressLevel', 'SleepQuality', 'StudyHoursPerWeek', 
                       'AcademicEngagement', 'SymptomFrequency_Last7Days']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    return preprocessor

def train_models(X_train, y_train, preprocessor):
    """Train multiple models and return the best one"""
    
    # Define models and parameters for grid search
    models = {
        'random_forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5]
            }
        },
        'logistic_regression': {
            'model': LogisticRegression(random_state=42),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['liblinear']
            }
        },
        'svm': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf']
            }
        }
    }
    
    best_model = None
    best_score = 0
    best_model_name = ""
    results = {}
    
    for name, config in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', config['model'])
        ])
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, 
            config['params'], 
            cv=5, 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store results
        results[name] = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_
        }
        
        print(f"Best ROC-AUC for {name}: {grid_search.best_score_:.4f}")
        
        # Update best model
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with ROC-AUC: {best_score:.4f}")
    
    return best_model, results

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test set"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    return roc_auc

def main():
    """Main training function"""
    print("Loading and preparing data...")
    X, y = prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Positive class ratio: {y.mean():.3f}")
    
    # Create preprocessor
    preprocessor = create_preprocessor()
    
    # Train models
    print("\nTraining models...")
    best_model, results = train_models(X_train, y_train, preprocessor)
    
    # Evaluate best model
    print("\nEvaluating best model on test set...")
    test_score = evaluate_model(best_model, X_test, y_test)
    
    # Save model
    Path("models").mkdir(exist_ok=True)
    
    with open("models/model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    # Save results
    with open("models/training_results.json", "w") as f:
        json.dump({
            'test_roc_auc': test_score,
            'model_results': results
        }, f, indent=2)
    
    # Save feature names
    feature_names = {
        'categorical': ['Gender', 'Course', 'YearOfStudy'],
        'numeric': ['Age', 'CGPA', 'StudyStressLevel', 'SleepQuality', 
                   'StudyHoursPerWeek', 'AcademicEngagement', 'SymptomFrequency_Last7Days']
    }
    
    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    
    print("\nModel and artifacts saved successfully!")

if __name__ == "__main__":
    main()