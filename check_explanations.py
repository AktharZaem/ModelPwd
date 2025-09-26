import json
from pathlib import Path
from collections import defaultdict
import sys

ANS_PATH = Path(
    "/Users/mohamedakthar/Desktop/Model Training/Paassword_Managment/ModelPwd/ModelPwd/answer_sheetpwd.json")
EXPL_PATH = Path(
    "/Users/mohamedakthar/Desktop/Model Training/Paassword_Managment/ModelPwd/ModelPwd/ExplanationBank.JSON")


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_entries(obj, out):
    # Recursively walk lists/dicts and collect dicts that have questionId and option
    if isinstance(obj, list):
        for item in obj:
            collect_entries(item, out)
    elif isinstance(obj, dict):
        if "questionId" in obj and "option" in obj:
            q = obj["questionId"]
            o = obj["option"]
            out[(q, o)] += 1
        # still traverse values for nested structures
        for v in obj.values():
            collect_entries(v, out)


def main():
    try:
        answers = load_json(ANS_PATH)
        expl_raw = load_json(EXPL_PATH)
    except Exception as e:
        print("Error loading JSON files:", e)
        sys.exit(2)

    expl_map = defaultdict(int)
    collect_entries(expl_raw, expl_map)

    missing = []
    unexpected_advanced = []

    questions = answers.get("questions", [])
    for i, qobj in enumerate(questions):
        qid = f"Q{i+1}"
        options = qobj.get("options", [])
        for j, opt in enumerate(options):
            opt_letter = chr(ord("A") + j)
            level = opt.get("level", "").strip()
            key = (qid, opt_letter)
            has_expl = expl_map.get(key, 0) > 0

            if level.lower() == "advanced":
                if has_expl:
                    unexpected_advanced.append(
                        {"questionId": qid, "option": opt_letter, "count": expl_map[key]})
            else:
                if not has_expl:
                    missing.append(
                        {"questionId": qid, "option": opt_letter, "level": level})

    # Print concise report
    if not missing and not unexpected_advanced:
        print("OK: All non-Advanced options have at least one explanation and Advanced options have no explanations.")
        return

    if missing:
        print("Missing explanations for non-Advanced options:")
        for m in missing:
            print(
                f" - {m['questionId']} option {m['option']} (level={m['level']}) is MISSING explanation(s).")

    if unexpected_advanced:
        print("\nUnexpected explanations found for Advanced options (should be none):")
        for u in unexpected_advanced:
            print(
                f" - {u['questionId']} option {u['option']} has {u['count']} explanation(s) (should be 0).")


if __name__ == "__main__":
    main()
