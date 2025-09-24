import json
import joblib
import pandas as pd
import numpy as np
from knowledge_enhancer import KnowledgeEnhancer


class PasswordSecurityTester:
    def __init__(self):
        self.answer_sheet = None
        self.questions_data = None
        self.model = None
        self.feature_names = None
        self.enhancer = KnowledgeEnhancer()
        self.load_components()

    def load_components(self):
        """Load trained model and answer sheet"""
        try:
            # Load answer sheet and parse the nested structure
            with open('answer_sheetpwd.json', 'r') as f:
                data = json.load(f)

            # Parse the nested JSON structure to match model_trainer format
            self.answer_sheet = {}
            self.questions_data = []

            if 'questions' in data and isinstance(data['questions'], list):
                for q_item in data['questions']:
                    question_text = q_item['question']
                    options_dict = {}

                    for option in q_item['options']:
                        options_dict[option['text']] = {
                            'weight': option['marks'],
                            'level': option['level']
                        }

                    self.answer_sheet[question_text] = options_dict
                    self.questions_data.append(q_item)

            # Load trained model
            self.model = joblib.load('password_security_model.pkl')
            self.feature_names = joblib.load('feature_names.pkl')

            print("Components loaded successfully!")
            print(f"Loaded {len(self.questions_data)} questions for quiz")

        except FileNotFoundError as e:
            print(f"Error loading components: {e}")
            print("Please run model_trainer.py first to train the model")

    def conduct_quiz(self):
        """Conduct interactive quiz with user"""
        print("\n=== Password Security Awareness Quiz ===")
        print("Please answer the following 10 questions about password management.\n")

        user_responses = {}
        user_scores = {}

        for i, q_item in enumerate(self.questions_data, 1):
            question = q_item['question']
            options = q_item['options']

            print(f"Question {i}: {question}")
            print("\nOptions:")

            # Display options
            for j, option in enumerate(options, 1):
                print(f"{j}. {option['text']}")

            # Get user input
            while True:
                try:
                    choice = int(
                        input(f"\nEnter your choice (1-{len(options)}): "))
                    if 1 <= choice <= len(options):
                        selected_option = options[choice - 1]
                        selected_answer = selected_option['text']

                        user_responses[question] = selected_answer

                        # Get score and level for this answer
                        user_scores[question] = {
                            'answer': selected_answer,
                            'score': selected_option['marks'],
                            'level': selected_option['level']
                        }
                        break
                    else:
                        print("Please enter a valid choice!")
                except ValueError:
                    print("Please enter a valid number!")

            print("-" * 50)

        return user_responses, user_scores

    def calculate_results(self, user_scores):
        """Calculate overall results and recommendations"""
        total_score = sum(score_info['score']
                          for score_info in user_scores.values())
        # Assuming max 10 points per question
        max_possible_score = len(user_scores) * 10
        percentage = (total_score / max_possible_score) * 100

        # Determine overall level
        if percentage >= 75:
            overall_level = 'Expert'
        elif percentage >= 50:
            overall_level = 'Intermediate'
        elif percentage >= 25:
            overall_level = 'Basic'
        else:
            overall_level = 'Beginner'

        return total_score, percentage, overall_level

    def provide_feedback(self, user_scores, overall_level, percentage):
        """Provide detailed feedback and recommendations"""
        print("\n" + "="*60)
        print("QUIZ RESULTS & PERSONALIZED FEEDBACK")
        print("="*60)

        total_score = sum(score_info['score']
                          for score_info in user_scores.values())
        print(f"Total Score: {total_score}/100")
        print(f"Percentage: {percentage:.1f}%")
        print(f"Overall Security Level: {overall_level}")

        if percentage >= 75:
            print("\n🎉 Congratulations! You're in the SAFE ZONE!")
            print("Your password security awareness is excellent.")
        else:
            print(f"\n⚠️ You need improvement in password security awareness.")

        print("\n" + "-"*60)
        print("DETAILED ANALYSIS BY QUESTION:")
        print("-"*60)

        improvement_areas = []

        for i, (question, score_info) in enumerate(user_scores.items(), 1):
            level = score_info['level']
            score = score_info['score']

            print(f"\nQuestion {i}: {question}")
            print(f"Your Answer Level: {level.upper()} ({score}/10 points)")

            if score < 10:  # Not perfect answer
                improvement_areas.append({
                    'question': question,
                    'current_level': level,
                    'score': score
                })

                print(f"💡 Enhancement Opportunity:")
                recommendation = self.enhancer.get_enhancement_advice(
                    question, level)
                print(f"   {recommendation}")

        # Overall recommendations
        if improvement_areas:
            print("\n" + "="*60)
            print("PRIORITY IMPROVEMENT AREAS:")
            print("="*60)

            # Sort by score (lowest first)
            improvement_areas.sort(key=lambda x: x['score'])

            for area in improvement_areas[:3]:  # Top 3 priority areas
                print(f"\n🎯 Priority: {area['question']}")
                print(f"   Current Level: {area['current_level']}")
                enhanced_advice = self.enhancer.get_detailed_guidance(
                    area['question'], area['current_level']
                )
                print(f"   📚 Learning Path: {enhanced_advice}")

    def run_assessment(self):
        """Run complete assessment process"""
        if not self.model or not self.answer_sheet:
            print(
                "Error: Model or answer sheet not loaded. Please train the model first.")
            return

        # Conduct quiz
        user_responses, user_scores = self.conduct_quiz()

        # Calculate results
        total_score, percentage, overall_level = self.calculate_results(
            user_scores)

        # Provide feedback
        self.provide_feedback(user_scores, overall_level, percentage)

        # Save user results
        user_data = {
            'responses': user_responses,
            'scores': user_scores,
            'total_score': total_score,
            'percentage': percentage,
            'overall_level': overall_level
        }

        with open('user_assessment_results.json', 'w') as f:
            json.dump(user_data, f, indent=2)

        print(f"\n📄 Results saved to 'user_assessment_results.json'")


if __name__ == "__main__":
    tester = PasswordSecurityTester()
    tester.run_assessment()
