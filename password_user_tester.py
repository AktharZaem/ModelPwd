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
        self.gemini_api_key = os.getenv(
            'GEMINI_API_KEY') or "AIzaSyDuDJ5uyh3DBAjEFTHaCz-g25fH7hp72Yc"
        self.explanation_bank = []            # <-- added
        self.profiles = {}                    # <-- added (from answer sheet)
        # <-- added (collected at runtime)
        self.user_profile = {}
        self.load_components()

    def load_components(self):
        """Load trained model and answer sheet"""
        try:
            # Load answer sheet and parse the nested structure
            with open('answer_sheetpwd.json', 'r') as f:
                data = json.load(f)

            self.answer_sheet = {}
            self.questions_data = []

            # store profiles if available
            self.profiles = data.get('profiles', {})

            if 'questions' in data and isinstance(data['questions'], list):
                for q_item in data['questions']:
                    question_text = q_item['question']
                    options_dict = {}

                    for opt in q_item['options']:
                        # option['label'] expected to exist (A/B/...) after updated JSON
                        options_dict[opt['text']] = {
                            'weight': opt['marks'],
                            'level': opt['level'],
                            # store label if provided
                            'label': opt.get('label')
                        }

                    self.answer_sheet[question_text] = options_dict
                    self.questions_data.append(q_item)

            # Load trained model
            self.model = joblib.load('password_model.pkl')
            self.feature_names = joblib.load('password_feature_names.pkl')

            # Try to load ExplanationBank.json if present
            try:
                with open('ExplanationBank.json', 'r') as ef:
                    raw_bank = json.load(ef)
                    # normalize/flatten the content to a list of dicts
                    self.explanation_bank = self._normalize_explanation_bank(raw_bank)
            except FileNotFoundError:
                self.explanation_bank = []

            print("Password components loaded successfully!")
            print(f"Loaded {len(self.questions_data)} questions for quiz")

        except FileNotFoundError as e:
            print(f"Error loading components: {e}")
            print("Please run password_model_trainer.py first to train the model")

    # helper to normalize ExplanationBank content
    def _normalize_explanation_bank(self, raw):
        """Return a flat list of dict entries from various plausible JSON shapes."""
        entries = []

        def _flatten(obj):
            if obj is None:
                return
            if isinstance(obj, dict):
                # if it contains a top-level list of entries, flatten that first
                for k in ('entries', 'explanations', 'items', 'data'):
                    if isinstance(obj.get(k), list):
                        for it in obj.get(k):
                            _flatten(it)
                        return
                # if dict looks like entry (has questionId or explanation), add it
                if 'questionId' in obj or 'explanation' in obj or 'option' in obj:
                    entries.append(obj)
                    return
                # otherwise, flatten values
                for v in obj.values():
                    _flatten(v)
                return
            if isinstance(obj, list):
                for item in obj:
                    _flatten(item)
                return
            # ignore other types

        _flatten(raw)
        # keep only dicts
        return [e for e in entries if isinstance(e, dict)]

    def conduct_quiz(self):
        """Conduct interactive quiz with user"""
        print("\n=== Password Management Security Awareness Quiz ===")
        print("Please answer the following 10 questions about password security.\n")

        # Collect user profile (gender, proficiency, education) before quiz
        self.user_profile = {}
        # Provide options if available in profiles, else allow free input
        genders = self.profiles.get('gender', [])
        proficiencies = self.profiles.get('proficiency', [])
        educations = self.profiles.get('education', [])

        def ask_choice(prompt, options):
            if options:
                print(prompt)
                for i, opt in enumerate(options, 1):
                    print(f" {i}. {opt}")
                while True:
                    resp = input(
                        f"Select (1-{len(options)}) or type a value: ").strip()
                    if resp.isdigit():
                        idx = int(resp)
                        if 1 <= idx <= len(options):
                            return options[idx - 1]
                    elif resp:
                        return resp
            else:
                resp = input(f"{prompt} ").strip()
                return resp if resp else ""

        self.user_profile['gender'] = ask_choice(
            "Select your gender:", genders) or "Unknown"
        self.user_profile['proficiency'] = ask_choice(
            "Select your proficiency:", proficiencies) or "Unknown"
        self.user_profile['education'] = ask_choice(
            "Select your education level:", educations) or "Unknown"

        print("\nCollected profile:", self.user_profile)
        print("\nStarting quiz...\n")

        user_responses = {}
        user_scores = {}

        for i, q_item in enumerate(self.questions_data, 1):
            question = q_item['question']
            options = q_item['options']

            print(f"Question {i}: {question}")
            print("\nOptions:")

            # Display options and assign labels A/B/C...
            for j, option in enumerate(options, 1):
                label = chr(ord('A') + j - 1)
                # attach label to option dict so it is available later
                option['label'] = label
                print(f"{label}. {option['text']}")

            # Get user input (accept label or number)
            while True:
                choice_raw = input(
                    f"\nEnter your choice (A-{chr(ord('A')+len(options)-1)} or 1-{len(options)}): ").strip()
                if not choice_raw:
                    print("Please enter a valid choice!")
                    continue

                # allow numeric selection
                if choice_raw.isdigit():
                    choice = int(choice_raw)
                    if 1 <= choice <= len(options):
                        selected_option = options[choice - 1]
                        break
                    else:
                        print("Please enter a valid numeric choice!")
                        continue

                # allow letter selection
                choice_upper = choice_raw.upper()
                if len(choice_upper) == 1 and 'A' <= choice_upper <= chr(ord('A') + len(options) - 1):
                    idx = ord(choice_upper) - ord('A')
                    selected_option = options[idx]
                    break

                print("Please enter a valid option (letter or number)!")

            selected_answer = selected_option['text']
            option_label = selected_option.get('label', None)
            question_id = q_item.get('questionId')

            user_responses[question] = selected_answer

            # Save score info along with questionId and option label
            user_scores[question] = {
                'answer': selected_answer,
                'score': selected_option['marks'],
                'level': selected_option['level'],
                'option_label': option_label,
                'questionId': question_id
            }

            print("-" * 50)

        return user_responses, user_scores

    def calculate_results(self, user_scores):
        """Calculate overall results and recommendations"""
        total_score = sum(score_info['score']
                          for score_info in user_scores.values())
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
        # keep as a wrapper to preserve backward compatibility, but not used directly now
        return self.get_explanation(question_text=question, question_id=None, option_label=None, current_level=current_level, overall_level=overall_level)

    # New helper: search ExplanationBank
    def find_explanation_from_bank(self, question_id, option_label, profile):
        """Try to find explanation in explanation_bank matching questionId + option + profile.
        Tries exact 3-attribute match first, then partial matches (2 attributes, 1 attribute)."""
        if not self.explanation_bank:
            return None

        # Score matches: exact match (3) > 2 attrs > 1 attr
        best_entry = None
        best_score = 0

        for entry in self.explanation_bank:
            # skip unexpected types
            if not isinstance(entry, dict):
                continue

            # Normalise possible question id keys
            entry_qid = entry.get('questionId') or entry.get('question_id') or entry.get('qid')
            if entry_qid is None:
                continue
            if str(entry_qid) != str(question_id):
                continue

            # option match required if present
            if 'option' in entry and entry['option'] is not None:
                if str(entry['option']).upper() != str(option_label).upper():
                    continue

            entry_profile = entry.get('profile', {})
            if not isinstance(entry_profile, dict):
                continue

            score = 0
            for key in ('gender', 'proficiency', 'education'):
                if entry_profile.get(key) and profile.get(key) and str(entry_profile.get(key)).lower() == str(profile.get(key)).lower():
                    score += 1

            # prefer higher score and require at least 1 attribute match
            if score > best_score:
                best_score = score
                best_entry = entry
                if best_score == 3:
                    break

        # return explanation if any match found (score>=1)
        if best_entry and best_score >= 1:
            return best_entry.get('explanation') or best_entry.get('text') or best_entry.get('detail')

        return None

    # New unified explanation getter that prefers ExplanationBank
    def get_explanation(self, question_text, question_id, option_label, current_level, overall_level):
        """Return explanation from ExplanationBank if available, else try Gemini, else fallback."""
        # Try ExplanationBank first (must have question_id and option_label)
        if question_id and option_label:
            expl = self.find_explanation_from_bank(
                question_id, option_label, self.user_profile)
            if expl:
                return f"\nPERSONALIZED EXPLANATION (from ExplanationBank):\n{expl}"

        # If ExplanationBank doesn't provide one, fall back to Gemini / generator
        # Use previous Gemini logic (slimmed down call) to avoid code duplication
        if not self.gemini_api_key:
            return self.get_detailed_explanation(question_text, current_level, overall_level)

        try:
            prompt = f"""
You are an expert cybersecurity educator specializing in password management and security.

CONTEXT:
- User's Question: "{question_text}"
- User's Current Answer Level: {current_level}
- User's Overall Knowledge Level: {overall_level}
- User Profile: {self.user_profile}

TASK:
Provide a personalized explanation to help this user understand password security concepts and advance to the next level.
"""
            model_names = ["gemini-1.5-flash", "gemini-1.5-pro",
                           "gemini-1.0-pro", "gemini-pro"]
            for model_name in model_names:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_api_key}"
                    headers = {"Content-Type": "application/json"}
                    data = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.7, "topK": 40, "topP": 0.95, "maxOutputTokens": 500}
                    }
                    response = requests.post(
                        url, headers=headers, json=data, timeout=30)
                    if response.status_code == 200:
                        result = response.json()
                        if 'candidates' in result and len(result['candidates']) > 0:
                            generated_text = result['candidates'][0]['content']['parts'][0]['text']
                            return f"\nPERSONALIZED EXPLANATION (from Gemini):\n{generated_text}"
                    else:
                        continue
                except Exception:
                    continue

            # fallback to local explanation if API failed
            return self.get_detailed_explanation(question_text, current_level, overall_level)

        except Exception:
            return self.get_detailed_explanation(question_text, current_level, overall_level)

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

        total_score = sum(score_info['score']
                          for score_info in user_scores.values())
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
                    'score': score,
                    'questionId': score_info.get('questionId'),
                    'option_label': score_info.get('option_label')
                })

                # Get explanation using ExplanationBank first
                ai_explanation = self.get_explanation(
                    question_text=question,
                    question_id=score_info.get('questionId'),
                    option_label=score_info.get('option_label'),
                    current_level=level,
                    overall_level=overall_level
                )
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
