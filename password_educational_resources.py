import json
from typing import List, Dict, Optional


class PasswordEducationalManager:
    def __init__(self):
        self.learning_resources = {
            'beginner': {
                'articles': [
                    {
                        'title': 'Password Security Basics',
                        'url': 'https://www.cisa.gov/secure-our-world/use-strong-passwords',
                        'description': 'CISA guide to creating and managing strong passwords'
                    },
                    {
                        'title': 'Password Manager Benefits',
                        'url': 'https://www.ftc.gov/consumer-advice/blog/2022/07/multi-factor-authentication-how-enable-it-and-why-you-should',
                        'description': 'FTC guide on password security and multi-factor authentication'
                    },
                    {
                        'title': 'Creating Strong Passwords',
                        'url': 'https://support.microsoft.com/en-us/windows/create-and-use-strong-passwords',
                        'description': 'Microsoft guide to creating strong passwords'
                    }
                ],
                'videos': [
                    {
                        'title': 'Password Security 101',
                        'platform': 'Educational Content',
                        'description': 'Basic concepts of password security and management'
                    }
                ]
            },
            'intermediate': {
                'articles': [
                    {
                        'title': 'Advanced Password Management',
                        'url': 'https://csrc.nist.gov/publications/detail/sp/800-63b/final',
                        'description': 'NIST authentication and lifecycle management guidelines'
                    },
                    {
                        'title': 'Enterprise Password Policies',
                        'url': 'https://www.sans.org/reading-room/whitepapers/authentication/passwords-policy-protecting-password-authentication-systems-486',
                        'description': 'SANS guide to password policy and authentication systems'
                    }
                ]
            },
            'advanced': {
                'articles': [
                    {
                        'title': 'Password Security Architecture',
                        'url': 'https://csrc.nist.gov/publications/detail/sp/800-118/draft',
                        'description': 'Advanced password security framework design'
                    },
                    {
                        'title': 'Zero Trust Authentication',
                        'url': 'https://www.cisa.gov/zero-trust-maturity-model',
                        'description': 'CISA zero trust architecture and implementation'
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
        """Generate personalized educational content for password security"""
        content = f"""
🔐 PERSONALIZED PASSWORD SECURITY LEARNING PLAN

Based on your assessment, here are key areas to focus on:

📚 PRIORITY AREAS FOR IMPROVEMENT:
"""

        area_explanations = {
            'password_creation': """
• STRONG PASSWORD CREATION: Master the art of creating unbreakable passwords
  - Use minimum 12-16 characters with complexity
  - Combine uppercase, lowercase, numbers, and special characters  
  - Avoid dictionary words, personal information, and patterns
  - Consider passphrases for better memorability and security
""",
            'password_managers': """
• PASSWORD MANAGER USAGE: Leverage technology for better security
  - Choose a reputable password manager (1Password, Bitwarden, etc.)
  - Generate unique passwords for every account
  - Enable secure password sharing features
  - Regularly backup and update your password vault
""",
            'two_factor_authentication': """
• MULTI-FACTOR AUTHENTICATION: Add extra layers of protection
  - Enable 2FA/MFA on all important accounts
  - Prefer authenticator apps over SMS when possible
  - Keep backup codes in a secure location
  - Understand different MFA methods and their security levels
""",
            'password_attacks': """
• UNDERSTANDING THREATS: Know your enemy to protect yourself
  - Learn about brute force, dictionary, and social engineering attacks
  - Understand how data breaches expose passwords
  - Monitor accounts for suspicious activity
  - Respond appropriately to security incidents
""",
            'account_security': """
• COMPREHENSIVE ACCOUNT SECURITY: Holistic protection approach
  - Regular security audits of all accounts
  - Monitor login attempts and account activity
  - Use secure email practices and recovery options
  - Implement proper session management
"""
        }

        for area in weak_areas:
            topic = self.map_area_to_topic(area)
            if topic in area_explanations:
                content += area_explanations[topic]

        content += f"""

🔧 PRACTICAL EXERCISES:
1. Audit all your current passwords and identify weak ones
2. Set up a password manager and migrate your passwords
3. Enable 2FA on your most important accounts (email, banking, social media)
4. Create a password security incident response plan

🌟 KNOWLEDGE LEVEL: {knowledge_level.upper()}
🎯 Goal: Achieve Expert level password security management!
"""
        return content

    def map_area_to_topic(self, area):
        """Map weak area to topic for educational content"""
        area_mapping = {
            'password_creation': 'password_creation',
            'password_strength': 'password_creation',
            'password_manager': 'password_managers',
            'two_factor': 'two_factor_authentication',
            '2fa': 'two_factor_authentication',
            'account_security': 'account_security',
            'password_policy': 'password_attacks'
        }
        
        for key, value in area_mapping.items():
            if key in area.lower():
                return value
        return 'password_creation'

    def get_interactive_tips(self) -> List[str]:
        """Get interactive password security tips"""
        return [
            "🔐 Use unique passwords for every account - never reuse passwords",
            "📏 Length matters more than complexity - aim for 12+ character passwords",
            "🛡️ Enable 2FA everywhere possible - it's your strongest defense",
            "📱 Use a reputable password manager - let technology help you",
            "🔍 Regularly audit your passwords - check for weak or reused ones",
            "⚠️ Monitor for data breaches - know when your passwords are compromised",
            "🔄 Update passwords immediately after any security incident"
        ]

    def display_resources(self, knowledge_level: str):
        """Display formatted educational resources"""
        resources = self.get_learning_resources(knowledge_level)

        print(f"\n🔐 PASSWORD SECURITY EDUCATIONAL RESOURCES - {knowledge_level.upper()} LEVEL")
        print("=" * 70)

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
        print("\n🔐 PASSWORD SECURITY EDUCATION CENTER")
        print("=" * 60)

        if quiz_score is not None:
            knowledge_level = self.assess_knowledge_level(quiz_score)
            print(f"\n📊 Your Assessment Score: {quiz_score:.1f}%")
            print(f"🎯 Knowledge Level: {knowledge_level.upper()}")

            if weak_areas:
                print(self.generate_personalized_content(weak_areas, knowledge_level))

            self.display_resources(knowledge_level)
        else:
            print("\n🔍 Select your current knowledge level:")
            print("1. Beginner - New to password security")
            print("2. Intermediate - Some password security awareness")
            print("3. Advanced - Strong password security background")

            choice = input("\nEnter your choice (1-3): ").strip()
            level_map = {'1': 'beginner', '2': 'intermediate', '3': 'advanced'}
            knowledge_level = level_map.get(choice, 'beginner')

            self.display_resources(knowledge_level)

        print(f"\n💡 QUICK TIPS:")
        tips = self.get_interactive_tips()
        for tip in tips[:4]:
            print(f"   {tip}")

        print(f"\n🎯 CHALLENGE: Improve your password security this week!")
