import pandas as pd
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
from difflib import SequenceMatcher
warnings.filterwarnings('ignore')


class PasswordModelTrainer:
    def __init__(self, dataset_path, answer_sheet_path):
        self.dataset_path = dataset_path
        self.answer_sheet_path = answer_sheet_path
        self.model = None
        self.answer_weights = None
        self.questions = None
        self.column_mappings = {}

    def similarity_score(self, a, b):
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def find_best_column_match(self, question, dataset_columns, threshold=0.2):
        """Find the best matching column for a question"""
        best_match = None
        best_score = 0

        for col in dataset_columns:
            # Direct keyword matching
            question_words = set(question.lower().split())
            column_words = set(col.lower().split())

            # Check for common keywords
            common_words = question_words.intersection(column_words)
            keyword_score = len(common_words) / \
                max(len(question_words), len(column_words))

            # Overall similarity
            similarity = self.similarity_score(question, col)

            # Combined score (prioritize keyword matching)
            combined_score = (keyword_score * 0.7) + (similarity * 0.3)

            if combined_score > best_score and combined_score > threshold:
                best_score = combined_score
                best_match = col

        return best_match, best_score

    def create_question_column_mapping(self):
        """Create mapping between answer sheet questions and dataset columns"""
        # Get all dataset columns
        all_columns = list(self.df.columns)

        # Specific demographic columns to exclude as specified by user
        specific_demographic_columns = [
            'Respondent_ID',
            'Timestamp',
            'Select Your Age',
            'Select Your Gender',
            'Select Your Education level',
            'IT proficiency at the'
        ]

        # Additional demographic keywords for safety
        demographic_keywords = [
            'respondent', 'id', 'timestamp', 'age', 'gender', 'education',
            'proficiency', 'name', 'email', 'phone', 'date', 'time'
        ]

        # Filter out demographic columns
        question_columns = []
        demographic_columns = []

        for col in all_columns:
            # First check specific columns to exclude
            if col in specific_demographic_columns:
                demographic_columns.append(col)
            else:
                # Then check for keyword matches
                col_lower = col.lower()
                is_demographic = any(
                    keyword in col_lower for keyword in demographic_keywords)

                if is_demographic:
                    demographic_columns.append(col)
                else:
                    question_columns.append(col)

        print(f"\n📊 Column Analysis:")
        print(f"   Total columns: {len(all_columns)}")
        print(f"   Demographic columns (excluded): {len(demographic_columns)}")
        if demographic_columns:
            for col in demographic_columns:
                print(f"      - {col}")
        print(f"   Question columns (available): {len(question_columns)}")
        if question_columns:
            for col in question_columns:
                print(f"      - {col}")

        print("\n🔍 Creating intelligent question-column mapping...")

        # First, try exact matches
        for question in self.questions:
            if question in question_columns:
                self.column_mappings[question] = question
                print(f"✅ Exact match: '{question}'")

        # For unmatched questions, find best similarity matches
        unmatched_questions = [
            q for q in self.questions if q not in self.column_mappings]
        available_columns = [
            col for col in question_columns if col not in self.column_mappings.values()]

        print(
            f"\n🔗 Mapping {len(unmatched_questions)} unmatched questions to {len(available_columns)} available columns:")

        for question in unmatched_questions:
            best_match, score = self.find_best_column_match(
                question, available_columns)

            if best_match:
                self.column_mappings[question] = best_match
                # Remove to avoid double mapping
                available_columns.remove(best_match)
                print(
                    f"   '{question[:50]}...' → '{best_match}' (similarity: {score:.2f})")
            else:
                print(f"   ❌ No suitable match for: '{question[:50]}...'")

        print(
            f"\n📊 Mapping Summary: {len(self.column_mappings)} out of {len(self.questions)} questions mapped")

        # Show final mappings
        if self.column_mappings:
            print(f"\n✅ Final Question-Column Mappings:")
            for i, (question, column) in enumerate(self.column_mappings.items(), 1):
                print(f"   {i}. Q: '{question[:40]}...' → C: '{column}'")

        return len(self.column_mappings) > 0

    def normalize_answer_text(self, answer):
        """Normalize answer text for better matching"""
        if pd.isna(answer):
            return ""

        # Convert to string and clean
        answer = str(answer).strip().lower()

        # Handle common variations
        replacements = {
            'and': '&',
            ',': ' ',
            '  ': ' ',
            'password': 'pwd',
        }

        for old, new in replacements.items():
            answer = answer.replace(old, new)

        return answer.strip()

    def find_best_answer_match(self, user_answer, available_options):
        """Find the best matching option for a user's answer"""
        normalized_answer = self.normalize_answer_text(user_answer)
        best_match = None
        best_score = 0

        for option_text, weight_info in available_options.items():
            normalized_option = self.normalize_answer_text(option_text)

            # Try different matching strategies
            scores = []

            # Exact match
            if normalized_answer == normalized_option:
                return option_text, weight_info

            # Substring matching
            if normalized_answer in normalized_option or normalized_option in normalized_answer:
                scores.append(0.8)

            # Word-based matching
            answer_words = set(normalized_answer.split())
            option_words = set(normalized_option.split())
            if answer_words and option_words:
                word_overlap = len(answer_words.intersection(
                    option_words)) / len(answer_words.union(option_words))
                scores.append(word_overlap * 0.6)

            # Similarity matching
            similarity = self.similarity_score(
                normalized_answer, normalized_option)
            scores.append(similarity * 0.4)

            combined_score = max(scores) if scores else 0

            if combined_score > best_score and combined_score > 0.5:
                best_score = combined_score
                best_match = (option_text, weight_info)

        return best_match if best_match else (None, {'weight': 0, 'level': 'wrong'})

    def load_answer_sheet(self):
        """Load weighted answers from JSON file"""
        with open(self.answer_sheet_path, 'r') as f:
            data = json.load(f)

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
        print(
            f"Loaded {len(self.questions)} password management questions from answer sheet")

    def load_dataset(self):
        """Load and preprocess the dataset"""
        self.df = pd.read_csv(self.dataset_path)
        print(f"\nOriginal dataset shape: {self.df.shape}")
        print(f"Original columns: {list(self.df.columns)}")

        # Don't remove specific columns - let the mapping process handle filtering
        print(f"\nDataset loaded successfully - will filter columns during mapping process")
        return self.df

    def calculate_user_scores(self):
        """Calculate scores for each user based on weighted answers"""
        # Create question-column mapping first
        if not self.create_question_column_mapping():
            raise ValueError("Could not create any question-column mappings!")

        scores = []
        detailed_scores = []

        print(
            f"\n🧮 Calculating scores for {len(self.df)} users using {len(self.column_mappings)} mapped questions...")

        for index, row in self.df.iterrows():
            user_score = 0
            user_details = {}

            print(f"\n   📝 User {index+1} responses:")

            for question, column in self.column_mappings.items():
                user_answer = str(row[column]).strip(
                ) if pd.notna(row[column]) else ""
                question_weights = self.answer_weights[question]

                # Find best matching answer
                matched_option, weight_info = self.find_best_answer_match(
                    user_answer, question_weights)

                score = weight_info['weight']
                level = weight_info['level']

                if matched_option:
                    print(
                        f"      '{user_answer}' → '{matched_option[:30]}...' ({score} pts, {level})")
                else:
                    print(
                        f"      '{user_answer}' → No match found ({score} pts, {level})")

                user_score += score
                user_details[question] = {
                    'answer': user_answer,
                    'matched_option': matched_option,
                    'score': score,
                    'level': level,
                    'column_used': column
                }

            scores.append(user_score)
            detailed_scores.append(user_details)
            print(f"      📊 User {index+1} Total Score: {user_score}")

        self.df['total_score'] = scores

        # Calculate max possible score based on mapped questions
        max_possible_score = 0
        for question in self.column_mappings.keys():
            question_weights = self.answer_weights[question]
            max_weight = max(info['weight']
                             for info in question_weights.values())
            max_possible_score += max_weight

        print(f"\n📊 Scoring Summary:")
        print(f"   Questions used for scoring: {len(self.column_mappings)}")
        print(f"   Max possible score: {max_possible_score}")

        # Calculate percentage based on actual max score
        self.df['percentage'] = (
            np.array(scores) / max_possible_score) * 100 if max_possible_score > 0 else 0
        self.detailed_scores = detailed_scores

        print(
            f"   Score range: {min(scores)} - {max(scores)} (Avg: {np.mean(scores):.1f})")
        print(
            f"   Percentage range: {self.df['percentage'].min():.1f}% - {self.df['percentage'].max():.1f}%")

        return scores, detailed_scores

    def classify_awareness_level(self):
        """Classify users into password security awareness levels"""
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

        print("\nPassword Security Awareness level distribution:")
        print(self.df['awareness_level'].value_counts())

        return self.df['awareness_level']

    def prepare_features(self):
        """Prepare features for ML training using only mapped columns"""
        feature_columns = []
        mapped_columns = list(self.column_mappings.values())

        print(
            f"\n🔧 Preparing ML features from {len(mapped_columns)} mapped columns...")

        for i, column in enumerate(mapped_columns):
            print(f"   Processing column: '{column}'")
            unique_answers = self.df[column].unique()
            print(
                f"      Unique answers: {len(unique_answers)} → {list(unique_answers)}")

            # Create dummy variables for each unique answer
            dummies = pd.get_dummies(self.df[column], prefix=f"Q_{i}")
            print(f"      Created features: {list(dummies.columns)}")

            feature_columns.extend(dummies.columns)
            self.df = pd.concat([self.df, dummies], axis=1)

        print(f"\n✅ Total ML features created: {len(feature_columns)}")
        print(f"   Feature names: {feature_columns[:10]}..." if len(
            feature_columns) > 10 else f"   Feature names: {feature_columns}")

        if len(feature_columns) == 0:
            raise ValueError(
                "No features could be created! Check question-column mappings.")

        X = self.df[feature_columns]
        y = self.df['awareness_level']

        print(f"\n📊 ML Dataset Summary:")
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Target variable shape: {y.shape}")
        print(f"   Target classes: {y.unique()}")

        return X, y

    def train_model(self):
        """Train the Decision Tree model for password management"""
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

        if X.empty or len(X) == 0:
            raise ValueError("Feature matrix is empty!")

        if len(y.unique()) < 2:
            print(f"Warning: Only {len(y.unique())} unique classes found")
            if len(y.unique()) == 1:
                print("Creating artificial class variation for training...")
                # Add some variation by slightly modifying scores
                # Change first row to create variation
                self.df.loc[0, 'awareness_level'] = 'Basic'
                y = self.df['awareness_level']

        print("Training password management model...")

        # Handle case where we might not have enough data for train/test split
        if len(X) < 4:  # Need at least 4 samples for split with stratify
            print("⚠️ Small dataset detected. Using simple split without stratification.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

        self.model = DecisionTreeClassifier(
            random_state=42, max_depth=10, min_samples_split=2, min_samples_leaf=1)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Password Management Model Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Save model and feature names
        joblib.dump(self.model, 'password_model.pkl')
        joblib.dump(X.columns.tolist(), 'password_feature_names.pkl')

        # Save column mappings for later use
        with open('password_column_mappings.json', 'w') as f:
            json.dump(self.column_mappings, f, indent=2)

        print("Model saved as 'password_model.pkl'")
        print("Column mappings saved as 'password_column_mappings.json'")

        return self.model, accuracy


if __name__ == "__main__":
    trainer = PasswordModelTrainer(
        dataset_path='password_form.csv',  # Updated to use correct file name
        answer_sheet_path='answer_sheetpwd.json'  # Updated to use correct file name
    )

    model, accuracy = trainer.train_model()
