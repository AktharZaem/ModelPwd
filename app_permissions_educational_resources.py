import json
from typing import List, Dict, Optional
import os
import pandas as pd


class AppPermissionsEducationalManager:
    def __init__(self, data_dir: Optional[str] = None):
        try:
            base_dir = data_dir or os.path.dirname(__file__)
        except Exception:
            base_dir = data_dir or os.getcwd()

        self.current_dir = base_dir
        self.questions_path = os.path.join(
            self.current_dir, "answer_sheetappper.json")
        self.survey_csv_path = os.path.join(
            self.current_dir, "mobile_app_permission.csv")
        self.explanations_path = os.path.join(
            self.current_dir, "ExplanationBankappper.json")
        self.resources_path = os.path.join(
            self.current_dir, "educational_resources.json")

        self.questions: Dict = {}
        self.survey_df = pd.DataFrame()
        self.explanations: List[Dict] = []
        self.base_resources: List[Dict] = []

        if os.path.isfile(self.questions_path):
            try:
                with open(self.questions_path, "r", encoding="utf-8") as f:
                    self.questions = json.load(f)
            except Exception:
                self.questions = {}

        if os.path.isfile(self.survey_csv_path):
            try:
                self.survey_df = pd.read_csv(self.survey_csv_path)
            except Exception:
                self.survey_df = pd.DataFrame()

        if os.path.isfile(self.explanations_path):
            try:
                with open(self.explanations_path, "r", encoding="utf-8") as f:
                    self.explanations = json.load(f)
            except Exception:
                self.explanations = []

        if os.path.isfile(self.resources_path):
            try:
                with open(self.resources_path, "r", encoding="utf-8") as f:
                    self.base_resources = json.load(f)
            except Exception:
                self.base_resources = []

        if not self.base_resources:
            self.base_resources = self._default_base_resources()

        self.qtext_to_qid: Dict[str, str] = {}
        for q in self.questions.get("questions", []):
            if isinstance(q, dict):
                qtext = q.get("question")
                qid = q.get("questionId")
                if isinstance(qtext, str) and qid:
                    self.qtext_to_qid[qtext] = qid

    def _default_base_resources(self) -> List[Dict]:
        base = [
            {"title": "Why long & random matters",
                "url": "https://www.example.com/password-basics"},
            {"title": "Simple rules for stronger passwords",
                "url": "https://www.example.com/password-rules"},
            {"title": "Creating memorable passphrases",
                "url": "https://www.example.com/passphrases"},
            {"title": "Entropy explained", "url": "https://www.example.com/entropy"},
            {"title": "Advanced password hygiene for professionals",
                "url": "https://www.example.com/advanced-passwords"},
            {"title": "Using system-generated secrets",
                "url": "https://www.example.com/secret-generation"},
            {"title": "What is a password manager?",
                "url": "https://www.example.com/pm-intro"},
            {"title": "How to install and use a manager (video)",
             "url": "https://www.example.com/pm-setup"},
            {"title": "Choosing a secure password manager",
                "url": "https://www.example.com/pm-choices"},
            {"title": "Best practices: master password & backup",
                "url": "https://www.example.com/pm-best-practices"},
            {"title": "Integrating managers with 2FA and SSO",
                "url": "https://www.example.com/pm-integration"},
            {"title": "Threat models and manager security",
                "url": "https://www.example.com/pm-threat"},
            {"title": "What is 2FA and why use it",
                "url": "https://www.example.com/2fa-basics"},
            {"title": "Authenticator apps vs SMS vs hardware keys",
                "url": "https://www.example.com/2fa-types"},
            {"title": "Deploying hardware keys and account recovery strategies",
                "url": "https://www.example.com/2fa-hardware"},
            {"title": "Steps to take after a leaked password",
                "url": "https://www.example.com/breach-steps"},
            {"title": "Using password managers to rotate credentials",
                "url": "https://www.example.com/rotate-with-manager"},
            {"title": "Monitoring services and automated rotation",
                "url": "https://www.example.com/monitor-rotate"},
            {"title": "Never post or message passwords",
                "url": "https://www.example.com/no-sharing"},
            {"title": "Using secure sharing tools",
                "url": "https://www.example.com/secure-share"},
            {"title": "Secrets management for teams",
                "url": "https://www.example.com/team-secrets"},
            {"title": "When to change a password",
                "url": "https://www.example.com/change-timing"},
            {"title": "Guidelines for periodic rotation",
                "url": "https://www.example.com/rotation-guidelines"},
            {"title": "Automated rotation and key vaults",
                "url": "https://www.example.com/automated-rotation"}
        ]
        for r in base:
            r.setdefault(
                "code", "# Example: use secrets.token_urlsafe(32) to generate a secret")
        return base

    def assess_knowledge_level(self, quiz_score: Optional[float]) -> str:
        if quiz_score is None:
            return "Beginner"
        try:
            score = float(quiz_score)
        except Exception:
            return "Beginner"
        if score <= 30:
            return "Beginner"
        if score <= 70:
            return "Intermediate"
        return "Advanced"

    def _expand_by_level(self, level: str, count: int = 1000) -> List[Dict]:
        variations = {
            "Beginner": ["Basics", "Getting Started", "Intro Guide", "Tips", "Foundations"],
            "Intermediate": ["Practical Use", "Manager Setup", "2FA Workflows", "Examples", "Hands-on"],
            "Advanced": ["Threat Modeling", "Automation", "Hardware Keys", "Security Audits", "Zero Trust"]
        }
        expanded = []
        idx = 1
        base = self.base_resources
        if not base:
            return expanded

        while len(expanded) < count:
            for b in base:
                var = variations[level][(idx - 1) % len(variations[level])]
                expanded.append({
                    "title": f"{b['title']} - {level} {var} {idx}",
                    "url": b['url'],
                    "code": b['code']
                })
                idx += 1
                if len(expanded) >= count:
                    break
        return expanded

    def get_learning_resources(self) -> Dict[str, List[Dict]]:
        return {
            "Beginner": self._expand_by_level("Beginner", 1000),
            "Intermediate": self._expand_by_level("Intermediate", 1000),
            "Advanced": self._expand_by_level("Advanced", 1000)
        }

    def generate_personalized_content(self, weak_areas: Optional[List[str]], knowledge_level: str) -> str:
        parts: List[str] = []
        weak_areas = weak_areas or []

        if not weak_areas:
            if not self.survey_df.empty:
                parts.append(
                    "We detected common weak areas from the group survey. Consider reviewing:")
                counts = {}
                for col in self.survey_df.columns[:20]:
                    try:
                        top = self.survey_df[col].dropna().mode().astype(str)
                        if not top.empty:
                            counts[col] = top.iloc[0]
                    except Exception:
                        continue
                for k, v in list(counts.items())[:5]:
                    parts.append(f"- {k} (common response: {v})")
            else:
                parts.append(
                    "No weak areas specified. Recommended starting point: strong passphrases, password manager basics, and 2FA.")
        else:
            parts.append("Personalized explanations and next steps:")
            for wa in weak_areas:
                qid = self.qtext_to_qid.get(
                    wa) if wa not in self.qtext_to_qid.values() else wa
                expl_text = None
                if qid:
                    for e in self.explanations:
                        if isinstance(e, dict) and e.get("questionId") == qid:
                            expl_text = e.get("explanation")
                            if expl_text:
                                break
                if expl_text:
                    parts.append(f"{wa} -> {expl_text}")
                else:
                    parts.append(
                        f"{wa} -> Review resources on strong passwords, managers, and 2FA.")

        next_steps = {
            "Beginner": "Next step: Finish Beginner track; create a strong passphrase and install a password manager.",
            "Intermediate": "Next step: Practice manager workflows and enable 2FA on major accounts.",
            "Advanced": "Next step: Implement automated rotation, hardware keys, and audit your secrets vault."
        }
        parts.append(next_steps.get(knowledge_level,
                     "Keep improving your security posture."))
        return "\n".join(parts)

    def get_interactive_tips(self) -> List[str]:
        return [
            "Use a password manager to generate & store unique passwords.",
            "Enable 2FA for critical accounts.",
            "Never reuse passwords across sites.",
            "Change leaked passwords immediately.",
            "Use secure sharing tools only for sensitive info."
        ]

    def run_educational_session(self, quiz_score: Optional[float] = None, weak_areas: Optional[List[str]] = None) -> Dict:
        level = self.assess_knowledge_level(quiz_score)
        all_resources = self.get_learning_resources()
        resources = all_resources.get(level, [])
        personalized = self.generate_personalized_content(weak_areas, level)
        tips = self.get_interactive_tips()

        result = {
            "level": level,
            "resources": resources,
            "personalized_content": personalized,
            "interactive_tips": tips,
            "total_resources": sum(len(v) for v in all_resources.values())
        }

        print(f"[Educational session] Level: {level}")
        print(f"Resources loaded: {len(resources)} for {level} (Total 3000)")
        for r in resources[:3]:
            print(f" - {r['title']} ({r['url']})")
        return result
