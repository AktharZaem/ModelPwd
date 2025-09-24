# Password Security Awareness Assessment System

This system trains a Machine Learning model to assess Generation Z's cybersecurity awareness specifically for password management and provides personalized feedback for improvement.

## Features

- **ML Model Training**: Uses Decision Tree classifier to analyze password security awareness
- **Weighted Scoring**: Implements 4-level scoring system (Advanced: 10, Intermediate: 5, Basic: 3, Wrong: 0)
- **Interactive Assessment**: Users can take a 10-question quiz
- **Personalized Feedback**: Provides detailed analysis and improvement recommendations
- **Knowledge Enhancement**: Suggests learning resources and next steps

## Files Structure

- `main.py` - Main execution script
- `model_trainer.py` - ML model training system
- `user_tester.py` - Interactive quiz and assessment
- `knowledge_enhancer.py` - Personalized recommendation system
- `answer_sheetpwd.json` - Weighted answer sheet for 10 questions
- `password_form.csv` - Training dataset (your 200 user responses)

## Setup

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have these files in the directory:
   - `password_form.csv` (your dataset)
   - `answer_sheetpwd.json` (provided)

## Usage

1. **Train the Model**:
   ```bash
   python main.py
   # Select option 1
   ```

2. **Take Assessment**:
   ```bash
   python main.py
   # Select option 2
   ```

## Assessment Levels

- **Expert (75%+)**: Safe zone - excellent security awareness
- **Intermediate (50-74%)**: Good awareness, some improvements needed
- **Basic (25-49%)**: Basic understanding, significant improvement needed
- **Beginner (<25%)**: Requires comprehensive security education

## Output Files

- `password_security_model.pkl` - Trained ML model
- `feature_names.pkl` - Feature names for the model
- `user_assessment_results.json` - Individual assessment results

## Customization

You can modify:
- Questions and weights in `answer_sheetpwd.json`
- Enhancement recommendations in `knowledge_enhancer.py`
- Scoring thresholds in `user_tester.py`
