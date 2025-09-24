import json
from urllib.parse import quote


class PasswordKnowledgeEnhancer:
    def __init__(self):
        self.enhancement_database = self.load_enhancement_database()

    def load_enhancement_database(self):
        """Load pre-defined enhancement recommendations for password security"""
        return {
            "password_creation": {
                "basic": "Learn how to create strong, unique passwords using length and complexity",
                "intermediate": "Understand advanced password creation techniques and passphrases",
                "advanced": "Master enterprise-level password policies and generation algorithms"
            },
            "password_managers": {
                "basic": "Understand what password managers are and why they're important",
                "intermediate": "Learn to choose and use password managers effectively",
                "advanced": "Implement enterprise password management solutions and policies"
            },
            "two_factor_authentication": {
                "basic": "Learn the basics of two-factor authentication (2FA) and its importance",
                "intermediate": "Understand different 2FA methods and their security implications",
                "advanced": "Implement advanced multi-factor authentication strategies"
            },
            "password_attacks": {
                "basic": "Understand common password attack methods and how to prevent them",
                "intermediate": "Learn about sophisticated attack vectors and defense strategies",
                "advanced": "Study advanced threat modeling and password security architecture"
            },
            "account_security": {
                "basic": "Learn about account security basics and password hygiene",
                "intermediate": "Understand account monitoring and breach response procedures",
                "advanced": "Implement comprehensive account security frameworks"
            },
            "password_policies": {
                "basic": "Understand basic password policy requirements and compliance",
                "intermediate": "Learn to develop effective organizational password policies",
                "advanced": "Design enterprise-grade password governance and risk management"
            }
        }

    def get_enhancement_advice(self, question, current_level):
        """Get basic enhancement advice for a question"""
        topic = self.map_question_to_topic(question)

        if topic in self.enhancement_database:
            if current_level in self.enhancement_database[topic]:
                return self.enhancement_database[topic][current_level]

        return f"Continue learning about {topic} to improve your password security awareness."

    def get_detailed_guidance(self, question, current_level):
        """Get detailed learning guidance with suggested resources"""
        topic = self.map_question_to_topic(question)
        next_level = self.get_next_level(current_level)

        guidance = f"To advance from {current_level} to {next_level} level in {topic}:\n"

        if current_level == "wrong" or current_level == "basic":
            guidance += f"   • Start with password security fundamentals\n"
            guidance += f"   • Practice creating strong passwords\n"
            guidance += f"   • Learn about password managers\n"
        elif current_level == "intermediate":
            guidance += f"   • Explore advanced {topic} concepts\n"
            guidance += f"   • Study security frameworks and standards\n"
            guidance += f"   • Consider cybersecurity certifications\n"

        # Add search suggestions
        search_terms = self.generate_search_terms(topic, current_level)
        guidance += f"   • Recommended searches: {', '.join(search_terms)}"

        return guidance

    def map_question_to_topic(self, question):
        """Map question text to password security topic"""
        question_lower = question.lower()

        if any(word in question_lower for word in ['create', 'strong', 'complex', 'length']):
            return 'password_creation'
        elif any(word in question_lower for word in ['manager', 'vault', 'store', 'remember']):
            return 'password_managers'
        elif any(word in question_lower for word in ['2fa', 'two-factor', 'authentication', 'verify']):
            return 'two_factor_authentication'
        elif any(word in question_lower for word in ['attack', 'hack', 'breach', 'compromise']):
            return 'password_attacks'
        elif any(word in question_lower for word in ['account', 'login', 'security']):
            return 'account_security'
        elif any(word in question_lower for word in ['policy', 'rule', 'requirement', 'standard']):
            return 'password_policies'
        else:
            return 'password_creation'

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
            'password_creation': ['strong password guide', 'password security best practices'],
            'password_managers': ['password manager tutorial', 'best password managers'],
            'two_factor_authentication': ['2FA setup guide', 'multi-factor authentication'],
            'password_attacks': ['password security threats', 'cybersecurity protection'],
            'account_security': ['account security tips', 'login security best practices'],
            'password_policies': ['password policy guidelines', 'enterprise password standards']
        }

        level_modifiers = {
            'basic': ['beginner guide', 'basics tutorial'],
            'intermediate': ['advanced guide', 'best practices'],
            'advanced': ['enterprise security', 'professional implementation']
        }

        terms = base_terms.get(topic, ['password security basics'])
        modifiers = level_modifiers.get(level, ['tutorial'])

        enhanced_terms = []
        for term in terms:
            enhanced_terms.append(f"{term} {modifiers[0]}")

        return enhanced_terms[:3]

    def generate_learning_path(self, user_scores):
        """Generate complete learning path based on user performance"""
        weak_areas = []
        for question, score_info in user_scores.items():
            if score_info['score'] < 7:
                weak_areas.append({
                    'topic': self.map_question_to_topic(question),
                    'level': score_info['level'],
                    'score': score_info['score']
                })

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
