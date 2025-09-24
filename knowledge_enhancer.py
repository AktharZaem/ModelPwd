import json
import requests
from urllib.parse import quote


class KnowledgeEnhancer:
    def __init__(self):
        self.enhancement_database = self.load_enhancement_database()

    def load_enhancement_database(self):
        """Load pre-defined enhancement recommendations"""
        return {
            "password_complexity": {
                "basic": "Learn about using uppercase, lowercase, numbers, and special characters",
                "intermediate": "Explore advanced password patterns and entropy concepts",
                "advanced": "Study password policy implementation and organizational standards"
            },
            "password_reuse": {
                "basic": "Understand why unique passwords are essential for each account",
                "intermediate": "Learn about password variation techniques and account categorization",
                "advanced": "Implement password rotation policies and breach response procedures"
            },
            "password_storage": {
                "basic": "Learn about password managers and secure storage methods",
                "intermediate": "Compare different password manager features and security models",
                "advanced": "Understand encryption methods and enterprise password management"
            },
            "two_factor_authentication": {
                "basic": "Learn what 2FA is and how to enable it on major platforms",
                "intermediate": "Compare TOTP, SMS, and hardware token security methods",
                "advanced": "Implement enterprise 2FA solutions and backup recovery methods"
            },
            "password_sharing": {
                "basic": "Understand the risks of sharing passwords and alternatives",
                "intermediate": "Learn secure credential sharing methods for teams",
                "advanced": "Implement role-based access control and credential management"
            }
        }

    def get_enhancement_advice(self, question, current_level):
        """Get basic enhancement advice for a question"""
        # Map question to topic
        topic = self.map_question_to_topic(question)

        if topic in self.enhancement_database:
            if current_level in self.enhancement_database[topic]:
                return self.enhancement_database[topic][current_level]

        return f"Continue learning about {topic} to improve your security awareness."

    def get_detailed_guidance(self, question, current_level):
        """Get detailed learning guidance with suggested resources"""
        topic = self.map_question_to_topic(question)
        next_level = self.get_next_level(current_level)

        guidance = f"To advance from {current_level} to {next_level} level in {topic}:\n"

        # Generate specific learning recommendations
        if current_level == "wrong" or current_level == "basic":
            guidance += f"   • Start with fundamentals of {topic}\n"
            guidance += f"   • Practice basic security principles\n"
            guidance += f"   • Use trusted educational resources\n"
        elif current_level == "intermediate":
            guidance += f"   • Deepen your understanding of {topic}\n"
            guidance += f"   • Explore advanced techniques and best practices\n"
            guidance += f"   • Consider professional certifications\n"

        # Add Google search suggestions
        search_terms = self.generate_search_terms(topic, current_level)
        guidance += f"   • Recommended searches: {', '.join(search_terms)}"

        return guidance

    def map_question_to_topic(self, question):
        """Map question text to security topic"""
        question_lower = question.lower()

        if any(word in question_lower for word in ['password', 'strong', 'weak', 'complexity']):
            return 'password_complexity'
        elif any(word in question_lower for word in ['reuse', 'same', 'different']):
            return 'password_reuse'
        elif any(word in question_lower for word in ['store', 'save', 'manager', 'remember']):
            return 'password_storage'
        elif any(word in question_lower for word in ['2fa', 'two-factor', 'authentication', 'verify']):
            return 'two_factor_authentication'
        elif any(word in question_lower for word in ['share', 'sharing', 'tell', 'give']):
            return 'password_sharing'
        else:
            return 'general_password_security'

    def get_next_level(self, current_level):
        """Determine the next level to aim for"""
        level_progression = {
            'wrong': 'basic',
            'basic': 'intermediate',
            'intermediate': 'advanced',
            'advanced': 'expert'
        }
        return level_progression.get(current_level, 'advanced')

    def generate_search_terms(self, topic, level):
        """Generate relevant search terms for learning"""
        base_terms = {
            'password_complexity': ['password strength checker', 'strong password examples'],
            'password_reuse': ['password uniqueness importance', 'account security best practices'],
            'password_storage': ['password manager comparison', 'secure password storage'],
            'two_factor_authentication': ['enable 2FA tutorial', 'two factor authentication guide'],
            'password_sharing': ['secure credential sharing', 'team password management']
        }

        level_modifiers = {
            'basic': ['beginner guide', 'tutorial'],
            'intermediate': ['best practices', 'advanced guide'],
            'advanced': ['enterprise security', 'professional certification']
        }

        terms = base_terms.get(topic, ['password security'])
        modifiers = level_modifiers.get(level, ['tutorial'])

        # Combine terms with modifiers
        enhanced_terms = []
        for term in terms:
            enhanced_terms.append(f"{term} {modifiers[0]}")

        return enhanced_terms[:3]  # Return top 3 suggestions

    def get_google_search_url(self, query):
        """Generate Google search URL for a query"""
        encoded_query = quote(query)
        return f"https://www.google.com/search?q={encoded_query}"

    def generate_learning_path(self, user_scores):
        """Generate complete learning path based on user performance"""
        weak_areas = []
        for question, score_info in user_scores.items():
            if score_info['score'] < 7:  # Areas needing improvement
                weak_areas.append({
                    'topic': self.map_question_to_topic(question),
                    'level': score_info['level'],
                    'score': score_info['score']
                })

        # Sort by score (weakest first)
        weak_areas.sort(key=lambda x: x['score'])

        learning_path = []
        for area in weak_areas:
            path_item = {
                'topic': area['topic'],
                'current_level': area['level'],
                'target_level': self.get_next_level(area['level']),
                'resources': self.generate_search_terms(area['topic'], area['level']),
                'priority': 'High' if area['score'] < 3 else 'Medium'
            }
            learning_path.append(path_item)

        return learning_path
