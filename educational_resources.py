import json
import requests
from typing import List, Dict, Optional
import os


class EducationalResourceManager:
    def __init__(self):
        self.gemini_api_key = os.getenv(
            'GEMINI_API_KEY')  # Set this in environment
        self.learning_resources = {
            'beginner': {
                'articles': [
                    {
                        'title': 'Password Security Fundamentals',
                        'url': 'https://www.nist.gov/itl/applied-cybersecurity/tig/back-basics-authentication',
                        'description': 'NIST guidelines on authentication basics'
                    },
                    {
                        'title': 'Creating Strong Passwords',
                        'url': 'https://www.cisa.gov/secure-our-world/use-strong-passwords',
                        'description': 'CISA guide to strong password creation'
                    },
                    {
                        'title': 'Password Manager Benefits',
                        'url': 'https://www.ftc.gov/consumer-advice/blog/2022/03/your-apps-want-you-create-account-heres-how-protect-your-privacy-and-data',
                        'description': 'FTC guidance on password management'
                    }
                ],
                'videos': [
                    {
                        'title': 'Password Security 101',
                        'platform': 'Educational Content',
                        'description': 'Basic concepts of password security'
                    }
                ]
            },
            'intermediate': {
                'articles': [
                    {
                        'title': 'Multi-Factor Authentication Guide',
                        'url': 'https://www.cisa.gov/mfa',
                        'description': 'Comprehensive MFA implementation guide'
                    },
                    {
                        'title': 'Password Policies in Organizations',
                        'url': 'https://csrc.nist.gov/publications/detail/sp/800-63b/final',
                        'description': 'NIST Special Publication on authentication guidelines'
                    }
                ]
            },
            'advanced': {
                'articles': [
                    {
                        'title': 'Zero Trust Security Models',
                        'url': 'https://www.cisa.gov/zero-trust-maturity-model',
                        'description': 'Advanced security architecture concepts'
                    },
                    {
                        'title': 'Cryptographic Best Practices',
                        'url': 'https://csrc.nist.gov/projects/cryptographic-standards-and-guidelines',
                        'description': 'NIST cryptographic standards'
                    }
                ]
            }
        }

    def assess_knowledge_level(self, quiz_score: float) -> str:
        """Determine knowledge level based on quiz performance"""
        if quiz_score >= 80:
            return 'advanced'
        elif quiz_score >= 60:
            return 'intermediate'
        else:
            return 'beginner'

    def get_learning_resources(self, knowledge_level: str) -> Dict:
        """Get curated learning resources based on knowledge level"""
        return self.learning_resources.get(knowledge_level, self.learning_resources['beginner'])

    def generate_personalized_content(self, weak_areas: List[str], knowledge_level: str) -> str:
        """Generate personalized educational content using Gemini AI (simulated)"""
        # Simulated Gemini response - replace with actual API call when available
        content = f"""
🎓 PERSONALIZED LEARNING PLAN

Based on your assessment, here are key areas to focus on:

📚 PRIORITY AREAS FOR IMPROVEMENT:
"""

        area_explanations = {
            'password_length': """
• PASSWORD LENGTH: Longer passwords exponentially increase security
  - Minimum 12 characters recommended
  - Each additional character multiplies cracking time
  - Consider passphrases: "Coffee$Morning#2024" vs "Cf2$"
""",
            'character_variety': """
• CHARACTER DIVERSITY: Mix different character types
  - Uppercase letters (A-Z)
  - Lowercase letters (a-z)  
  - Numbers (0-9)
  - Special symbols (!@#$%^&*)
  - Example: "MyDog#Loves2Play!" combines all types
""",
            'common_patterns': """
• AVOID PREDICTABLE PATTERNS: Hackers know common substitutions
  - Don't use: "password123" or "P@ssw0rd"
  - Avoid: sequential numbers, keyboard patterns
  - Skip: dictionary words with simple substitutions
""",
            'personal_info': """
• PERSONAL INFORMATION: Never use identifiable data
  - Avoid: birthdays, names, addresses
  - Skip: social media information
  - Don't use: phone numbers or SSN parts
""",
            'reuse_prevention': """
• UNIQUE PASSWORDS: Each account needs its own password
  - Use password managers for unique generation
  - Consider themed variations for memorability
  - Implement different passwords for different risk levels
"""
        }

        for area in weak_areas:
            if area in area_explanations:
                content += area_explanations[area]

        content += f"""

🔧 PRACTICAL EXERCISES:
1. Create 3 strong passwords using different methods
2. Set up a reputable password manager
3. Enable 2FA on your most important accounts
4. Review and update your oldest passwords

🌟 KNOWLEDGE LEVEL: {knowledge_level.upper()}
Next assessment in 2 weeks to track your progress!
"""
        return content

    def get_interactive_tips(self) -> List[str]:
        """Get interactive security tips"""
        return [
            "💡 Try the 'diceware' method: roll dice to create random word combinations",
            "🎲 Use the 'first letter' technique: take first letters of a memorable sentence",
            "🔄 Implement the 'base + variation' system for multiple accounts",
            "📱 Practice identifying phishing attempts that steal passwords",
            "🛡️ Learn about password-less authentication methods"
        ]

    def display_resources(self, knowledge_level: str):
        """Display formatted educational resources"""
        resources = self.get_learning_resources(knowledge_level)

        print(f"\n📚 EDUCATIONAL RESOURCES - {knowledge_level.upper()} LEVEL")
        print("=" * 60)

        print("\n📖 RECOMMENDED ARTICLES:")
        for i, article in enumerate(resources.get('articles', []), 1):
            print(f"\n{i}. {article['title']}")
            print(f"   📝 {article['description']}")
            print(f"   🔗 {article['url']}")

        if 'videos' in resources:
            print(f"\n🎥 VIDEO RESOURCES:")
            for i, video in enumerate(resources.get('videos', []), 1):
                print(f"\n{i}. {video['title']}")
                print(f"   📝 {video['description']}")
                print(f"   📺 Platform: {video['platform']}")

    def run_educational_session(self, quiz_score: Optional[float] = None, weak_areas: Optional[List[str]] = None):
        """Run an interactive educational session"""
        print("\n🎓 PASSWORD SECURITY EDUCATION CENTER")
        print("=" * 50)

        if quiz_score is not None:
            knowledge_level = self.assess_knowledge_level(quiz_score)
            print(f"\n📊 Your Assessment Score: {quiz_score:.1f}%")
            print(f"🎯 Knowledge Level: {knowledge_level.upper()}")

            if weak_areas:
                print(
                    "\n" + self.generate_personalized_content(weak_areas, knowledge_level))

            self.display_resources(knowledge_level)
        else:
            print("\n🔍 Select your current knowledge level:")
            print("1. Beginner - New to password security")
            print("2. Intermediate - Some security awareness")
            print("3. Advanced - Strong security background")

            choice = input("\nEnter your choice (1-3): ").strip()
            level_map = {'1': 'beginner', '2': 'intermediate', '3': 'advanced'}
            knowledge_level = level_map.get(choice, 'beginner')

            self.display_resources(knowledge_level)

        print(f"\n💡 QUICK TIPS:")
        tips = self.get_interactive_tips()
        for tip in tips[:3]:  # Show first 3 tips
            print(f"   {tip}")

        print(f"\n🎯 CHALLENGE: Try implementing one new security practice this week!")
