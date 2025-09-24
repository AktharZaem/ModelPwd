import pandas as pd
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class PasswordSecurityModelTrainer:
    def __init__(self, dataset_path, answer_sheet_path):
        self.dataset_path = dataset_path
        self.answer_sheet_path = answer_sheet_path
        self.model = None
        self.answer_weights = None
        self.questions = None

    def load_answer_sheet(self):
        """Load weighted answers from JSON file"""
        with open(self.answer_sheet_path, 'r') as f:
            data = json.load(f)

        # Parse the nested JSON structure
        self.answer_weights = {}

        if 'questions' in data and isinstance(data['questions'], list):
            for q_item in data['questions']:
                question_text = q_item['question']
                options_dict = {}

                for option in q_item['options']:
                    options_dict[option['text']] = {
                        'weight': option['marks'],
                        'level': option['level']
                    }

                self.answer_weights[question_text] = options_dict

        self.questions = list(self.answer_weights.keys())
        print(f"Loaded {len(self.questions)} questions from answer sheet")
        print(f"Questions parsed:")
        for i, question in enumerate(self.questions, 1):
            print(f"  {i}. {question}")

    def load_dataset(self):
        """Load and preprocess the dataset"""
        self.df = pd.read_csv(self.dataset_path)
        print(f"\nOriginal dataset shape: {self.df.shape}")
        print(f"Original columns: {list(self.df.columns)}")

        # Define columns to remove (demographic data)
        columns_to_remove = [
            'Respondent_ID',
            'Timestamp',
            'Select Your Age',
            'Select Your Gender',
            'Select Your Education level',
            'IT proficiency at the'
        ]

        # Remove demographic columns
        columns_removed = []
        for col in columns_to_remove:
            if col in self.df.columns:
                self.df = self.df.drop(columns=[col])
                columns_removed.append(col)

        print(f"\nRemoved {len(columns_removed)} demographic columns:")
        for col in columns_removed:
            print(f"  - {col}")

        print(f"\nFiltered dataset shape: {self.df.shape}")
        print(f"Remaining columns: {list(self.df.columns)}")

        # Debug: Show which questions match remaining columns
        print("\nMatching questions to remaining columns:")
        matched_count = 0
        for question in self.questions:
            if question in self.df.columns:
                print(f"✅ '{question}' found in dataset")
                matched_count += 1
            else:
                print(f"❌ '{question}' NOT found in dataset")
                # Try to find similar column names
                similar_cols = [col for col in self.df.columns if any(
                    word.lower() in col.lower() for word in question.split()[:3])]
                if similar_cols:
                    print(f"   Similar columns: {similar_cols}")

        print(
            f"\nMatched {matched_count} out of {len(self.questions)} questions")
        return self.df

    def calculate_user_scores(self):
        """Calculate scores for each user based on weighted answers"""
        scores = []
        detailed_scores = []
        matched_questions = []

        # First, identify which questions actually exist in the dataset
        for question in self.questions:
            if question in self.df.columns:
                matched_questions.append(question)

        print(
            f"\nUsing {len(matched_questions)} matched questions for scoring")

        if len(matched_questions) == 0:
            raise ValueError(
                "No questions from answer sheet match dataset columns!")

        for index, row in self.df.iterrows():
            user_score = 0
            user_details = {}

            for question in matched_questions:
                user_answer = str(row[question]).strip()
                question_weights = self.answer_weights[question]

                # Find matching weight for user's answer
                score = 0
                level = 'wrong'
                for answer_option, weight_info in question_weights.items():
                    if user_answer.lower() == answer_option.lower():
                        score = weight_info['weight']
                        level = weight_info['level']
                        break

                user_score += score
                user_details[question] = {
                    'answer': user_answer,
                    'score': score,
                    'level': level
                }

            scores.append(user_score)
            detailed_scores.append(user_details)

        self.df['total_score'] = scores

        # Calculate max possible score based on matched questions
        max_possible_score = 0
        for question in matched_questions:
            question_weights = self.answer_weights[question]
            max_weight = max(info['weight']
                             for info in question_weights.values())
            max_possible_score += max_weight

        print(f"Max possible score: {max_possible_score}")

        # Calculate percentage based on actual max score
        self.df['percentage'] = (
            np.array(scores) / max_possible_score) * 100 if max_possible_score > 0 else 0
        self.detailed_scores = detailed_scores

        return scores, detailed_scores

    def classify_awareness_level(self):
        """Classify users into awareness levels"""
        def get_level(percentage):
            if percentage >= 75:
                return 'Expert'
            elif percentage >= 50:
                return 'Intermediate'
            elif percentage >= 25:
                return 'Basic'
            else:
                return 'Beginner'

        self.df['awareness_level'] = self.df['percentage'].apply(get_level)

        # Show distribution
        print("\nAwareness level distribution:")
        print(self.df['awareness_level'].value_counts())

        return self.df['awareness_level']

    def prepare_features(self):
        """Prepare features for ML training"""
        # Convert categorical answers to numerical features
        feature_columns = []

        # Only use questions that exist in the dataset
        matched_questions = [q for q in self.questions if q in self.df.columns]

        print(
            f"\nPreparing features from {len(matched_questions)} questions...")

        for question in matched_questions:
            # Create dummy variables for each question's answers
            print(f"Processing question: '{question[:50]}...'")
            unique_answers = self.df[question].unique()
            print(f"  Unique answers: {len(unique_answers)}")

            dummies = pd.get_dummies(
                self.df[question], prefix=f"Q_{len(feature_columns)}")
            print(f"  Created {len(dummies.columns)} dummy features")

            feature_columns.extend(dummies.columns)
            self.df = pd.concat([self.df, dummies], axis=1)

        print(f"\nTotal features created: {len(feature_columns)}")

        if len(feature_columns) == 0:
            raise ValueError(
                "No features could be created! Check if questions match dataset columns.")

        X = self.df[feature_columns]
        y = self.df['awareness_level']

        print(f"Feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        print(f"Target classes: {y.unique()}")

        return X, y

    def train_model(self):
        """Train the Decision Tree model"""
        print("Loading answer sheet...")
        self.load_answer_sheet()

        print("Loading dataset...")
        self.load_dataset()

        print("Calculating user scores...")
        self.calculate_user_scores()

        print("Classifying awareness levels...")
        self.classify_awareness_level()

        print("Preparing features...")
        X, y = self.prepare_features()

        # Validate that we have enough data
        if X.empty or len(X) == 0:
            raise ValueError("Feature matrix is empty!")

        if len(y.unique()) < 2:
            print(f"Warning: Only {len(y.unique())} unique classes found")
            if len(y.unique()) == 1:
                raise ValueError("Cannot train model with only one class!")

        print("Training model...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        self.model = DecisionTreeClassifier(
            random_state=42, max_depth=10, min_samples_split=5)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Save model and feature names
        joblib.dump(self.model, 'password_security_model.pkl')
        joblib.dump(X.columns.tolist(), 'feature_names.pkl')

        print("Model saved as 'password_security_model.pkl'")
        return self.model, accuracy


if __name__ == "__main__":
    trainer = PasswordSecurityModelTrainer(
        dataset_path='password_form.csv',
        answer_sheet_path='answer_sheetpwd.json'
    )

    model, accuracy = trainer.train_model()
