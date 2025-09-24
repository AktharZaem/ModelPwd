#!/usr/bin/env python3

import os
import sys
import traceback
import json

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def check_dependencies():
    """Check if required modules can be imported"""
    missing_modules = []

    try:
        from model_trainer import PasswordSecurityModelTrainer
        print("✅ model_trainer module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import model_trainer: {e}")
        missing_modules.append("model_trainer")

    try:
        from user_tester import PasswordSecurityTester
        print("✅ user_tester module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import user_tester: {e}")
        missing_modules.append("user_tester")

    return missing_modules


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
                    # Show first 5 subkeys
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
        'password_form.csv',
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
    print("Password Security Awareness Assessment System")
    print("=" * 50)

    # Check dependencies first
    print("\nChecking dependencies...")
    missing_modules = check_dependencies()

    if missing_modules:
        print(f"\n❌ Missing required modules: {', '.join(missing_modules)}")
        print("Please ensure all required Python files are in the same directory.")
        return

    # Import modules after checking
    try:
        from model_trainer import PasswordSecurityModelTrainer
        from user_tester import PasswordSecurityTester
    except ImportError as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return

    while True:
        print("\nSelect an option:")
        print("1. Train ML Model")
        print("2. Take Assessment Quiz")
        print("3. Check System Status")
        print("4. Check JSON Structure")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == '1':
            print("\n--- Training ML Model ---")
            check_files()

            # Check JSON structure first
            if not check_json_structure():
                print("❌ JSON structure check failed!")
                continue

            # Check if required files exist
            if not os.path.exists('password_form.csv'):
                print("Error: password_form.csv not found!")
                print("Please ensure the dataset file is in the current directory.")
                continue

            if not os.path.exists('answer_sheetpwd.json'):
                print("Error: answer_sheetpwd.json not found!")
                print("Please ensure the answer sheet file is in the current directory.")
                continue

            try:
                print("Initializing trainer...")
                trainer = PasswordSecurityModelTrainer(
                    dataset_path='password_form.csv',
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
            print("\n--- Password Security Assessment ---")

            # Check if model exists
            if not os.path.exists('password_security_model.pkl'):
                print("❌ Trained model not found!")
                print("Please train the model first (option 1).")
                continue

            try:
                print("Initializing tester...")
                tester = PasswordSecurityTester()
                tester.run_assessment()

            except Exception as e:
                print(f"❌ Error during assessment: {e}")
                traceback.print_exc()

        elif choice == '3':
            print("\n--- System Status ---")
            check_files()

            # Check for model file
            if os.path.exists('password_security_model.pkl'):
                size = os.path.getsize('password_security_model.pkl')
                print(f"✅ password_security_model.pkl (size: {size} bytes)")
            else:
                print("❌ password_security_model.pkl not found")

        elif choice == '4':
            print("\n--- JSON Structure Check ---")
            check_json_structure()

        elif choice == '5':
            print("\nThank you for using the Password Security Assessment System!")
            print("Stay secure! 🔒")
            break

        else:
            print("Invalid choice! Please enter 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
