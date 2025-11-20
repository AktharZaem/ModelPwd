import os  # added
import glob
import pandas as pd
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
import matplotlib.pyplot as plt  # Added for plotting
warnings.filterwarnings('ignore')

# Add optional import to reuse parsing from tester if available
try:
    from app_permissions_user_tester import AppPermissionsTester
except Exception:
    AppPermissionsTester = None


class AppPermissionsModelTrainer:
    def __init__(self, dataset_path, answer_sheet_path, assessment_results_path='app_permissions_assessment_results.json'):
        # Use absolute paths based on script directory for robustness
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(script_dir, os.path.basename(
            dataset_path)) if not os.path.isabs(dataset_path) else dataset_path
        self.answer_sheet_path = os.path.join(script_dir, os.path.basename(
            answer_sheet_path)) if not os.path.isabs(answer_sheet_path) else answer_sheet_path
        self.assessment_results_path = os.path.join(script_dir, os.path.basename(
            assessment_results_path)) if not os.path.isabs(assessment_results_path) else assessment_results_path
        self.model = None
        self.answer_weights = None
        self.questions = None

    def normalize_text(self, text):
        """Normalize text by lowercasing and keeping only alphanumeric characters for matching."""
        return ''.join(c.lower() for c in text if c.isalnum())

    def load_answer_sheet(self):
        """Load weighted answers from JSON file.
        Prefer to reuse AppPermissionsTester parsing if available so both trainer/tester share the same structure.
        """
        # If tester is available, try to reuse its parsed answer sheet
        if AppPermissionsTester is not None:
            try:
                tester = AppPermissionsTester()
                if getattr(tester, 'answer_sheet', None):
                    self.answer_weights = tester.answer_sheet
                    # questions_data may be available on tester; fall back to keys
                    if getattr(tester, 'questions_data', None):
                        self.questions = [
                            q.get('question') for q in tester.questions_data if q.get('question')]
                    else:
                        self.questions = list(self.answer_weights.keys())
                    # Create normalized mapping for flexible matching
                    self.normalized_questions = {
                        self.normalize_text(q): q for q in self.questions}
                    print(
                        "✅ Reused answer sheet parsing from app_permissions_user_tester.py")
                    print(
                        f"Loaded {len(self.questions)} app permissions questions (via tester)")
                    return
            except Exception as e:
                print(f"⚠️ Could not reuse tester parsing: {e}")

        # Fallback: parse answer sheet JSON directly
        try:
            with open(self.answer_sheet_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not open answer sheet '{self.answer_sheet_path}': {e}")

        self.answer_weights = {}

        if 'questions' in data and isinstance(data['questions'], list):
            for q_item in data['questions']:
                question_text = q_item.get(
                    'question') or q_item.get('questionText') or None
                if not question_text:
                    continue
                options_dict = {}

                for option in q_item.get('options', []):
                    # tolerate different key names
                    text = option.get('text') or option.get('label') or ''
                    marks = option.get('marks', option.get('score', 0))
                    level = option.get('level', 'basic')
                    options_dict[text] = {
                        'weight': marks,
                        'level': level
                    }

                self.answer_weights[question_text] = options_dict

        self.questions = list(self.answer_weights.keys())
        # Create normalized mapping for flexible matching
        self.normalized_questions = {
            self.normalize_text(q): q for q in self.questions}
        print(
            f"Loaded {len(self.questions)} app permissions questions from answer sheet")
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

        # Create normalized mapping for columns
        self.normalized_columns = {self.normalize_text(
            col): col for col in self.df.columns}

        # Debug: Show which questions match remaining columns using normalized matching
        print("\nMatching questions to remaining columns:")
        matched_count = 0
        for norm_q, orig_q in self.normalized_questions.items():
            if norm_q in self.normalized_columns:
                print(f"✅ '{orig_q}' found in dataset")
                matched_count += 1
            else:
                print(f"❌ '{orig_q}' NOT found in dataset")
                # Try to find similar column names (optional, for debugging)
                similar_cols = [col for col in self.df.columns if any(
                    word.lower() in col.lower() for word in orig_q.split()[:3])]
                if similar_cols:
                    print(f"   Similar columns: {similar_cols}")

        print(
            f"\nMatched {matched_count} out of {len(self.questions)} questions")
        return self.df

    def load_assessment_results(self):
        """Load and convert assessment results from JSON to DataFrame format.
        Accept multiple common formats:
         - file contains a single user result (as written by the tester)
         - file contains {'results': [...]} list
         - file contains {'assessments': [...]} or other similar shapes
        """
        # Try the explicitly provided path first
        data = None
        tried_paths = []
        try_paths = [self.assessment_results_path]

        # if the provided path is None or missing, search for common filenames
        try:
            if not self.assessment_results_path or not os.path.exists(self.assessment_results_path):
                candidates = glob.glob("app_permissions_assessment*.json") + \
                    glob.glob("app_permissions_assessment_database*.json")
                # ensure uniqueness and sensible order
                for c in candidates:
                    if c not in try_paths:
                        try_paths.append(c)

            for p in try_paths:
                if not p:
                    continue
                tried_paths.append(p)
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"✅ Loaded assessment results from: {p}")
                    break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"⚠️ Error reading '{p}': {e}")
                    continue

            if data is None:
                print(f"Assessment results not found. Tried: {tried_paths}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error searching for assessment results: {e}")
            return pd.DataFrame()

        # Normalize to a list of result dicts
        results = []
        if isinstance(data, dict):
            # common keys
            if 'results' in data and isinstance(data['results'], list):
                results = data['results']
            elif 'assessments' in data and isinstance(data['assessments'], list):
                results = data['assessments']
            else:
                # It may be a single user result dict (as saved by tester)
                # Heuristic: presence of 'profile' and 'responses' keys
                if 'profile' in data and 'responses' in data:
                    results = [data]
                else:
                    # try to detect list-like containers inside dict
                    for v in data.values():
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            results = v
                            break
        elif isinstance(data, list):
            results = data

        if not results:
            print("No assessment results found in JSON file.")
            return pd.DataFrame()

        print(f"Loading {len(results)} assessment results from JSON...")

        # Convert JSON results to DataFrame rows
        rows = []
        for result in results:
            row = {}
            # Add profile data
            profile = result.get('profile', result.get('user_profile', {}))
            row['gender'] = profile.get('gender', '')
            row['education_level'] = profile.get(
                'education') or profile.get('education_level', '')
            row['proficiency'] = profile.get('proficiency', '')

            # Add responses
            responses = result.get('responses') or result.get('answers') or {}
            for question, answer in responses.items():
                row[question] = answer

            # Add calculated fields (try multiple key variants)
            row['total_score'] = result.get(
                'total_score', result.get('score', 0))
            row['percentage'] = result.get(
                'percentage', result.get('percent', 0))
            row['awareness_level'] = result.get(
                'overall_level', result.get('awareness_level', 'Unknown'))

            rows.append(row)

        assessment_df = pd.DataFrame(rows)
        print(f"Assessment results DataFrame shape: {assessment_df.shape}")
        return assessment_df

    def combine_datasets(self):
        """Combine CSV dataset with assessment results"""
        csv_df = self.load_dataset()
        assessment_df = self.load_assessment_results()

        if assessment_df.empty:
            print("No assessment data to combine. Using only CSV data.")
            self.df = csv_df
        else:
            # Ensure consistent column names
            common_columns = set(csv_df.columns) & set(assessment_df.columns)
            print(f"Common columns between datasets: {len(common_columns)}")

            # Combine datasets
            self.df = pd.concat([csv_df, assessment_df], ignore_index=True)
            print(f"Combined dataset shape: {self.df.shape}")

        return self.df

    def calculate_user_scores(self):
        """Calculate scores for each user based on weighted answers"""
        scores = []
        detailed_scores = []
        # Use normalized matching to build matched_questions with original CSV column names
        matched_questions = [self.normalized_columns[norm_q]
                             for norm_q in self.normalized_questions if norm_q in self.normalized_columns]

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
                # Map CSV column back to JSON question for weights
                norm_q = self.normalize_text(question)
                json_question = self.normalized_questions.get(norm_q)
                if not json_question:
                    continue
                question_weights = self.answer_weights[json_question]

                # Find matching weight for user's answer
                score = 0
                level = 'wrong'
                for answer_option, weight_info in question_weights.items():
                    if user_answer.lower() == answer_option.lower():
                        score = weight_info['weight']
                        level = weight_info['level']
                        break

                user_score += score
                user_details[json_question] = {
                    'answer': user_answer,
                    'score': score,
                    'level': level
                }

            scores.append(user_score)
            detailed_scores.append(user_details)

        self.df['total_score'] = scores

        # Calculate max possible score based on matched questions
        max_possible_score = 0
        for norm_q in self.normalized_questions:
            if norm_q in self.normalized_columns:
                json_question = self.normalized_questions[norm_q]
                question_weights = self.answer_weights[json_question]
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
        """Classify users into app permissions awareness levels"""
        def get_level(percentage):
            if percentage >= 75:
                return 'Expert'
            elif percentage >= 50:
                return 'Intermediate'
            else:
                return 'Beginner'

        self.df['awareness_level'] = self.df['percentage'].apply(get_level)

        print("\nApp Permissions Awareness level distribution:")
        print(self.df['awareness_level'].value_counts())

        return self.df['awareness_level']

    def prepare_features(self):
        """Prepare features for ML training"""
        feature_columns = []
        # Use matched_questions from calculate_user_scores (ensure it's set)
        if not hasattr(self, 'matched_questions'):
            self.matched_questions = [self.normalized_columns[norm_q]
                                      for norm_q in self.normalized_questions if norm_q in self.normalized_columns]
        matched_questions = self.matched_questions

        print(
            f"\nPreparing features from {len(matched_questions)} questions...")

        for question in matched_questions:
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
        """Train the SVM model for app permissions"""
        print("Loading answer sheet...")
        self.load_answer_sheet()

        print("Loading and combining datasets...")
        self.combine_datasets()

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
                raise ValueError("Cannot train model with only one class!")

        print("Training app permissions SVM model...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Use SVM for multiclass classification with RBF kernel and default C=1.0
        self.model = SVC(random_state=42, kernel='rbf', C=1.0)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"SVM Model Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Generate and save plots
        self.generate_training_report(y_test, y_pred)

        # Save model and feature names
        joblib.dump(self.model, 'app_permissions_model.pkl')
        joblib.dump(X.columns.tolist(), 'app_permissions_feature_names.pkl')

        print("Model saved as 'app_permissions_model.pkl'")
        return self.model, accuracy

    def generate_training_report(self, y_test, y_pred):
        """Generate and save visual reports of training results"""
        # Plot 1: Awareness Level Distribution
        awareness_counts = self.df['awareness_level'].value_counts()
        plt.figure(figsize=(8, 6))
        awareness_counts.plot(kind='bar', color='skyblue')
        plt.title('Awareness Level Distribution')
        plt.xlabel('Awareness Level')
        plt.ylabel('Number of Users')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('awareness_level_distribution.png')
        plt.close()
        print("✅ Awareness level distribution plot saved as 'awareness_level_distribution.png'")

        # Plot 2: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=y_test.unique())
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm, cmap='Blues')
        fig.colorbar(cax)
        ax.set_xticks(range(len(y_test.unique())))
        ax.set_yticks(range(len(y_test.unique())))
        ax.set_xticklabels(y_test.unique(), rotation=45)
        ax.set_yticklabels(y_test.unique())
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # Add numbers to each cell
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        print("✅ Confusion matrix plot saved as 'confusion_matrix.png'")

        # Plot 3: Classification Metrics (Precision, Recall, F1-Score)
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=y_test.unique())
        labels = y_test.unique()
        x = np.arange(len(labels))
        width = 0.2
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, precision, width,
               label='Precision', color='lightcoral')
        ax.bar(x, recall, width, label='Recall', color='lightgreen')
        ax.bar(x + width, f1, width, label='F1-Score', color='lightblue')
        ax.set_xlabel('Classes')
        ax.set_title('Classification Metrics by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        plt.tight_layout()
        plt.savefig('classification_metrics.png')
        plt.close()
        print("✅ Classification metrics plot saved as 'classification_metrics.png'")

        # New Plot 4: Model Accuracy Bar Chart
        accuracy = accuracy_score(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Model Accuracy'], [accuracy * 100], color='green')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Model Accuracy')
        ax.set_ylim(0, 100)
        ax.text(0, accuracy * 100 + 1,
                f'{accuracy * 100:.2f}%', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig('model_accuracy_plot.png')
        plt.close()
        print("✅ Model accuracy plot saved as 'model_accuracy_plot.png'")

        # Save classification report to text file
        report = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        with open('classification_report.txt', 'w') as f:
            f.write("SVM Classification Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Overall Accuracy: {accuracy:.2f}\n\n")
            f.write(report)
        print("✅ Classification report saved as 'classification_report.txt'")

        print("\n📊 Training report files generated. You can download 'awareness_level_distribution.png', 'confusion_matrix.png', 'classification_metrics.png', 'model_accuracy_plot.png', and 'classification_report.txt'.")


if __name__ == "__main__":
    trainer = AppPermissionsModelTrainer(
        dataset_path='mobile_app_permission.csv',
        answer_sheet_path='answer_sheetappper.json'
    )

    model, accuracy = trainer.train_model()
