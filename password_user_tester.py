import json
import joblib
import pandas as pd
import numpy as np
import os
import requests
from password_knowledge_enhancer import PasswordKnowledgeEnhancer


class PasswordTester:
    def __init__(self):
        self.answer_sheet = None
        self.questions_data = None
        self.model = None
        self.feature_names = None
        self.enhancer = PasswordKnowledgeEnhancer()
        # Google API for enhanced learning
        self.gemini_api_key = os.getenv('GEMINI_API_KEY') or "AIzaSyDuDJ5uyh3DBAjEFTHaCz-g25fH7hp72Yc"
        self.load_components()

    def load_components(self):
        """Load trained model and answer sheet"""
        try:
            # Load answer sheet and parse the nested structure
            with open('answer_sheetpwd.json', 'r') as f:
                data = json.load(f)

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
            self.model = joblib.load('password_model.pkl')
            self.feature_names = joblib.load('password_feature_names.pkl')

            print("Password components loaded successfully!")
            print(f"Loaded {len(self.questions_data)} questions for quiz")

        except FileNotFoundError as e:
            print(f"Error loading components: {e}")
            print("Please run password_model_trainer.py first to train the model")

    def conduct_quiz(self):
        """Conduct interactive quiz with user"""
        print("\n=== Password Management Security Awareness Quiz ===")
        print("Please answer the following 10 questions about password security.\n")

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
                    choice = int(input(f"\nEnter your choice (1-{len(options)}): "))
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
        total_score = sum(score_info['score'] for score_info in user_scores.values())
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

    def get_gemini_explanation(self, question, current_level, overall_level):
        """Get personalized explanation from Gemini API"""
        if not self.gemini_api_key:
            return self.get_detailed_explanation(question, current_level, overall_level)

        try:
            prompt = f"""
You are an expert cybersecurity educator specializing in password management and security.

CONTEXT:
- User's Question: "{question}"
- User's Current Answer Level: {current_level}
- User's Overall Knowledge Level: {overall_level}

TASK:
Provide a personalized explanation to help this user understand password security concepts and advance to the next level.

GUIDELINES:
- If user is at "wrong" level, explain password basics very simply
- If user is at "basic" level, provide more detailed password security concepts
- If user is at "intermediate" level, give advanced password management practices
- If user is at "advanced" level, provide expert-level password security insights

FORMAT:
- Use emojis and clear structure
- Include practical examples
- Explain WHY this matters for their security
- Give actionable next steps
- Keep it engaging and educational
- Maximum 300 words

Please provide a comprehensive explanation that will help them improve from their current level to the next level.
"""

            model_names = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"]

            for model_name in model_names:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_api_key}"

                    headers = {"Content-Type": "application/json"}
                    data = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.7,
                            "topK": 40,
                            "topP": 0.95,
                            "maxOutputTokens": 500,
                        }
                    }

                    print(f"🔐 Analyzing your password security knowledge...")
                    response = requests.post(url, headers=headers, json=data, timeout=30)

                    if response.status_code == 200:
                        result = response.json()
                        if 'candidates' in result and len(result['candidates']) > 0:
                            generated_text = result['candidates'][0]['content']['parts'][0]['text']
                            return f"\nPERSONALIZED EXPLANATION:\n{generated_text}"
                    else:
                        continue

                except Exception:
                    continue

            return self.get_detailed_explanation(question, current_level, overall_level)

        except Exception as e:
            return self.get_detailed_explanation(question, current_level, overall_level)

    def get_detailed_explanation(self, question, current_level, overall_level):
        """Get detailed explanation based on question and user's knowledge level"""
        # Fallback explanations for common password questions
        return f"""
🔐 PASSWORD SECURITY LEARNING OPPORTUNITY:
Your current level: {current_level.upper()}
Overall knowledge: {overall_level.upper()}

📚 KEY CONCEPTS TO UNDERSTAND:
• Password complexity and length requirements
• Unique passwords for different accounts
• Password manager benefits and usage
• Two-factor authentication importance
• Common password attacks and prevention

🎯 NEXT STEPS TO IMPROVE:
• Practice creating strong passwords
• Learn about password managers
• Understand security best practices
• Research current password standards
"""

    def provide_feedback(self, user_scores, overall_level, percentage):
        """Provide detailed feedback and recommendations"""
        print("\n" + "="*60)
        print("PASSWORD MANAGEMENT QUIZ RESULTS & PERSONALIZED FEEDBACK")
        print("="*60)

        total_score = sum(score_info['score'] for score_info in user_scores.values())
        print(f"Total Score: {total_score}/100")
        print(f"Percentage: {percentage:.1f}%")
        print(f"Overall Password Security Level: {overall_level}")

        # Provide level-specific encouragement
        if percentage >= 75:
            print("\n🎉 Congratulations! You're in the SAFE ZONE!")
            print("Your password security knowledge is excellent.")
        elif percentage >= 50:
            print("\n📈 Good Progress! You're at INTERMEDIATE level!")
            print("You have solid foundation but room for improvement.")
        elif percentage >= 25:
            print("\n📚 You're at BASIC level - Learning Time!")
            print("Password security can be complex, but you're on the right track!")
        else:
            print("\n🌱 You're just getting started - BEGINNER level!")
            print("Perfect opportunity to learn strong password practices!")

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

                # Get AI-generated explanation
                ai_explanation = self.get_gemini_explanation(question, level, overall_level)
                print(ai_explanation)

        # Overall recommendations
        if improvement_areas:
            print("\n" + "="*60)
            print("PRIORITY IMPROVEMENT AREAS:")
            print("="*60)

            improvement_areas.sort(key=lambda x: x['score'])

            for area in improvement_areas[:3]:
                print(f"\n🎯 Priority: {area['question']}")
                print(f"   Current Level: {area['current_level'].upper()}")

                enhanced_advice = self.enhancer.get_detailed_guidance(
                    area['question'], area['current_level']
                )
                print(f"   📚 Learning Path: {enhanced_advice}")

        print("\n" + "="*60)
        if overall_level.lower() == 'beginner':
            print("🌟 REMEMBER: Strong passwords are your first line of defense!")
        elif overall_level.lower() == 'basic':
            print("🚀 YOU'RE IMPROVING! Keep learning password best practices!")
        elif overall_level.lower() == 'intermediate':
            print("🎯 ALMOST EXPERT! Focus on advanced password security!")
        else:
            print("🏆 EXCELLENT! You understand password security well!")

    def run_assessment(self):
        """Run complete assessment process"""
        if not self.model or not self.answer_sheet:
            print("Error: Model or answer sheet not loaded. Please train the model first.")
            return

        # Conduct quiz
        user_responses, user_scores = self.conduct_quiz()

        # Calculate results
        total_score, percentage, overall_level = self.calculate_results(user_scores)

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

        with open('password_assessment_results.json', 'w') as f:
            json.dump(user_data, f, indent=2)

        print(f"\n📄 Results saved to 'password_assessment_results.json'")

        return {
            'score': percentage,
            'weak_areas': [question for question, score_info in user_scores.items() if score_info['score'] < 7]
        }


if __name__ == "__main__":
    tester = PasswordTester()
    tester.run_assessment()
