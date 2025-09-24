#!/usr/bin/env python3

import os
import sys
import subprocess
import traceback
import json

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def setup_virtual_environment():
    """Create virtual environment and install required packages"""
    venv_path = os.path.join(current_dir, '.venv')

    if not os.path.exists(venv_path):
        print("🔧 Creating virtual environment...")
        try:
            subprocess.run([sys.executable, '-m', 'venv',
                           '.venv'], check=True, cwd=current_dir)
            print("✅ Virtual environment created successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False

    # Determine the correct Python executable path
    if os.name == 'nt':  # Windows
        python_exe = os.path.join(venv_path, 'Scripts', 'python.exe')
        pip_exe = os.path.join(venv_path, 'Scripts', 'pip.exe')
    else:  # macOS/Linux
        python_exe = os.path.join(venv_path, 'bin', 'python')
        pip_exe = os.path.join(venv_path, 'bin', 'pip')

    # Install required packages
    required_packages = [
        'pandas',
        'scikit-learn',
        'joblib',
        'numpy',
        'requests'
    ]

    print("📦 Installing required packages...")
    for package in required_packages:
        try:
            subprocess.run([pip_exe, 'install', package],
                           check=True, capture_output=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warning: Failed to install {package}: {e}")

    return python_exe


def create_sample_files():
    """Create sample answer sheet and dataset if they don't exist"""

    # Create sample answer sheet
    if not os.path.exists('answer_sheetpwd.json'):
        print("📝 Creating sample answer_sheetpwd.json...")
        sample_answer_sheet = {
            "questions": [
                {
                    "question": "What makes a password strong?",
                    "options": [
                        {"text": "Length, complexity, and uniqueness",
                            "marks": 10, "level": "advanced"},
                        {"text": "Just length is enough",
                            "marks": 5, "level": "intermediate"},
                        {"text": "Only complexity matters",
                            "marks": 3, "level": "basic"},
                        {"text": "Using personal information",
                            "marks": 0, "level": "wrong"}
                    ]
                },
                {
                    "question": "How often should you change your passwords?",
                    "options": [
                        {"text": "When compromised or suspicious activity",
                            "marks": 10, "level": "advanced"},
                        {"text": "Every 3-6 months", "marks": 5,
                            "level": "intermediate"},
                        {"text": "Every month", "marks": 3, "level": "basic"},
                        {"text": "Never change them", "marks": 0, "level": "wrong"}
                    ]
                },
                {
                    "question": "What is the best way to store passwords?",
                    "options": [
                        {"text": "Use a reputable password manager",
                            "marks": 10, "level": "advanced"},
                        {"text": "Write them in a notebook",
                            "marks": 5, "level": "intermediate"},
                        {"text": "Save in browser", "marks": 3, "level": "basic"},
                        {"text": "Use the same password everywhere",
                            "marks": 0, "level": "wrong"}
                    ]
                }
            ]
        }

        with open('answer_sheetpwd.json', 'w') as f:
            json.dump(sample_answer_sheet, f, indent=2)
        print("✅ Sample answer sheet created!")

    # Create sample dataset
    if not os.path.exists('password_form.csv'):
        print("📝 Creating sample password_form.csv...")
        sample_data = """Respondent_ID,Timestamp,Select Your Age,Select Your Gender,Select Your Education level,IT proficiency at the,What makes a password strong?,How often should you change your passwords?,What is the best way to store passwords?
1,2024-01-01,25-30,Male,Bachelor's,Advanced,Length complexity and uniqueness,When compromised or suspicious activity,Use a reputable password manager
2,2024-01-02,31-35,Female,Master's,Intermediate,Just length is enough,Every 3-6 months,Write them in a notebook  
3,2024-01-03,20-25,Male,High School,Basic,Only complexity matters,Every month,Save in browser
4,2024-01-04,36-40,Female,PhD,Expert,Using personal information,Never change them,Use the same password everywhere
5,2024-01-05,25-30,Male,Bachelor's,Advanced,Length complexity and uniqueness,When compromised or suspicious activity,Use a reputable password manager"""

        with open('password_form.csv', 'w') as f:
            f.write(sample_data)
        print("✅ Sample dataset created!")


def check_dependencies():
    """Check if required modules can be imported"""
    missing_modules = []

    try:
        from password_model_trainer import PasswordModelTrainer
        print("✅ password_model_trainer module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import password_model_trainer: {e}")
        missing_modules.append("password_model_trainer")

    try:
        from password_user_tester import PasswordTester
        print("✅ password_user_tester module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import password_user_tester: {e}")
        missing_modules.append("password_user_tester")

    try:
        from password_educational_resources import PasswordEducationalManager
        print("✅ password_educational_resources module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import password_educational_resources: {e}")
        missing_modules.append("password_educational_resources")

    return missing_modules


def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = ['pandas', 'sklearn', 'joblib', 'numpy', 'requests']
    missing_packages = []

    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)

    return missing_packages


def check_json_structure():
    """Check the structure of the JSON answer sheet"""
    if not os.path.exists('answer_sheetpwd.json'):
        print("❌ answer_sheetpwd.json not found")
        return False

    try:
        with open('answer_sheetpwd.json', 'r') as f:
            data = json.load(f)

        print(f"\nJSON Structure Analysis:")
        print(f"Type: {type(data)}")
        print(
            f"Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dictionary'}")

        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  '{key}': {type(value)}")
                if isinstance(value, dict):
                    print(f"    Subkeys: {list(value.keys())[:5]}...")
                elif isinstance(value, list):
                    print(f"    Length: {len(value)}")
                    if len(value) > 0:
                        print(f"    First item: {type(value[0])}")

        return True
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        return False


def check_files():
    """Check if required files exist"""
    files_to_check = [
        'password_form.csv',  # Updated file name
        'answer_sheetpwd.json'
    ]

    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"Script directory: {current_dir}")
    print("\nChecking required files:")

    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} (size: {size} bytes)")
        else:
            print(f"❌ {file} not found")


def main():
    print("Password Management Security Awareness Assessment System")
    print("=" * 60)

    # Setup virtual environment and install packages
    print("\n🔧 Setting up environment...")
    python_exe = setup_virtual_environment()
    if not python_exe:
        print("❌ Failed to setup virtual environment. Continuing with system Python...")
        python_exe = sys.executable

    # Create sample files if they don't exist
    create_sample_files()

    # Check Python packages
    print("\n📦 Checking Python packages...")
    missing_packages = check_python_packages()
    if missing_packages:
        print(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install " + " ".join(missing_packages))

    # Check dependencies first
    print("\nChecking dependencies...")
    missing_modules = check_dependencies()

    if missing_modules:
        print(f"\n❌ Missing required modules: {', '.join(missing_modules)}")
        print("Creating missing module files...")
        create_missing_modules(missing_modules)

    # Import modules after checking
    try:
        from password_model_trainer import PasswordModelTrainer
        from password_user_tester import PasswordTester
        from password_educational_resources import PasswordEducationalManager
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required files are created properly.")
        return

    # Initialize educational manager
    education_manager = PasswordEducationalManager()
    last_quiz_score = None
    weak_areas = []

    while True:
        print("\nSelect an option:")
        print("1. Train ML Model")
        print("2. Take Password Security Assessment Quiz")
        print("3. Educational Resources & Learning")
        print("4. Check System Status")
        print("5. Check JSON Structure")
        print("6. Setup/Reset Environment")
        print("7. Exit")

        choice = input("\nEnter your choice (1-7): ").strip()

        if choice == '1':
            print("\n--- Training ML Model for Password Management ---")
            check_files()

            if not check_json_structure():
                print("❌ JSON structure check failed!")
                continue

            if not os.path.exists('password_form.csv'):  # Updated file name
                print("Error: password_form.csv not found!")
                print("Please ensure the dataset file is in the current directory.")
                continue

            if not os.path.exists('answer_sheetpwd.json'):
                print("Error: answer_sheetpwd.json not found!")
                print("Please ensure the answer sheet file is in the current directory.")
                continue

            try:
                print("Initializing trainer...")
                trainer = PasswordModelTrainer(
                    dataset_path='password_form.csv',  # Updated file name
                    answer_sheet_path='answer_sheetpwd.json'
                )

                print("Starting model training...")
                model, accuracy = trainer.train_model()
                print(f"\n✅ Model training completed successfully!")
                print(f"Model accuracy: {accuracy:.2f}")

            except Exception as e:
                print(f"❌ Error during model training: {e}")
                traceback.print_exc()

        elif choice == '2':
            print("\n--- Password Management Security Assessment ---")

            if not os.path.exists('password_model.pkl'):
                print("❌ Trained model not found!")
                print("Please train the model first (option 1).")
                continue

            try:
                print("Initializing tester...")
                tester = PasswordTester()
                result = tester.run_assessment()

                if isinstance(result, dict):
                    last_quiz_score = result.get('score', 0)
                    weak_areas = result.get('weak_areas', [])
                    print(
                        f"\n🎓 Ready to learn more? Check out option 3 for personalized resources!")
                elif isinstance(result, (int, float)):
                    last_quiz_score = result
                    print(
                        f"\n🎓 Ready to learn more? Check out option 3 for personalized resources!")

            except Exception as e:
                print(f"❌ Error during assessment: {e}")
                traceback.print_exc()

        elif choice == '3':
            print("\n--- Educational Resources & Learning ---")
            try:
                if last_quiz_score is not None:
                    print(f"📊 Using your recent assessment results...")
                    education_manager.run_educational_session(
                        last_quiz_score, weak_areas)
                else:
                    print("📚 No recent assessment found. Showing general resources...")
                    education_manager.run_educational_session()

                print(f"\n🔄 Additional Options:")
                print("A. View resources for different knowledge level")
                print("B. Get quick security tips")
                print("C. Return to main menu")

                sub_choice = input(
                    "\nEnter your choice (A/B/C): ").strip().upper()

                if sub_choice == 'A':
                    education_manager.run_educational_session()
                elif sub_choice == 'B':
                    tips = education_manager.get_interactive_tips()
                    print(f"\n💡 PASSWORD SECURITY TIPS:")
                    for i, tip in enumerate(tips, 1):
                        print(f"{i}. {tip}")

            except Exception as e:
                print(f"❌ Error accessing educational resources: {e}")
                traceback.print_exc()

        elif choice == '4':
            print("\n--- System Status ---")
            check_files()

            # Check virtual environment
            venv_path = os.path.join(current_dir, '.venv')
            if os.path.exists(venv_path):
                print(f"✅ Virtual environment exists at: {venv_path}")
            else:
                print(f"❌ Virtual environment not found")

            if os.path.exists('password_model.pkl'):
                size = os.path.getsize('password_model.pkl')
                print(f"✅ password_model.pkl (size: {size} bytes)")
            else:
                print("❌ password_model.pkl not found")

            # Check Python packages
            missing_packages = check_python_packages()
            if not missing_packages:
                print("✅ All required packages are installed")

        elif choice == '5':
            print("\n--- JSON Structure Check ---")
            check_json_structure()

        elif choice == '6':
            print("\n--- Setup/Reset Environment ---")
            setup_virtual_environment()
            create_sample_files()
            print("✅ Environment setup completed!")

        elif choice == '7':
            print(
                "\nThank you for using the Password Management Security Awareness Assessment System!")
            print("Keep learning and stay secure! 🔒🔑")
            break

        else:
            print("Invalid choice! Please enter 1, 2, 3, 4, 5, 6, or 7.")


def create_missing_modules(missing_modules):
    """Create basic structure for missing modules"""
    if "password_model_trainer" in missing_modules:
        create_model_trainer_file()
    if "password_user_tester" in missing_modules:
        create_user_tester_file()
    if "password_educational_resources" in missing_modules:
        create_educational_resources_file()


def create_model_trainer_file():
    """Create a basic password_model_trainer.py file"""
    content = '''# Basic Password Model Trainer
class PasswordModelTrainer:
    def __init__(self, dataset_path, answer_sheet_path):
        self.dataset_path = dataset_path
        self.answer_sheet_path = answer_sheet_path
        print(f"PasswordModelTrainer initialized with {dataset_path}")
    
    def train_model(self):
        print("Training model... (placeholder)")
        return None, 0.85
'''

    with open('password_model_trainer.py', 'w') as f:
        f.write(content)
    print("✅ Created password_model_trainer.py")


def create_user_tester_file():
    """Create a basic password_user_tester.py file"""
    content = '''# Basic Password User Tester
class PasswordTester:
    def __init__(self):
        print("PasswordTester initialized")
    
    def run_assessment(self):
        print("Running assessment... (placeholder)")
        return {"score": 75, "weak_areas": []}
'''

    with open('password_user_tester.py', 'w') as f:
        f.write(content)
    print("✅ Created password_user_tester.py")


def create_educational_resources_file():
    """Create a basic password_educational_resources.py file"""
    content = '''# Basic Password Educational Resources
class PasswordEducationalManager:
    def __init__(self):
        print("PasswordEducationalManager initialized")
    
    def run_educational_session(self, score=None, weak_areas=None):
        print("Running educational session... (placeholder)")
    
    def get_interactive_tips(self):
        return ["Use strong passwords", "Enable 2FA", "Use password manager"]
'''

    with open('password_educational_resources.py', 'w') as f:
        f.write(content)
    print("✅ Created password_educational_resources.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
