import json
from pathlib import Path


# ==============================
# PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "RSVQA"

IMAGE_DIR = DATASET_DIR / "Images_LR"

OUTPUT_DIR = DATASET_DIR / "prepared"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# FUNCTION TO LOAD JSON
# ==============================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================
# PREPARE ONE DATA SPLIT
# ==============================

def prepare_split(split_name, question_file, answer_file):

    print(f"\nPreparing {split_name}...")

    questions_data = load_json(question_file)
    answers_data = load_json(answer_file)

    # RSVQA JSON files contain lists under these keys
    questions = questions_data["questions"]
    answers = answers_data["answers"]

    # --------------------------------
    # Create answer lookup
    # --------------------------------

    answer_lookup = {}

    for answer in answers:
        answer_id = answer["id"]
        answer_lookup[answer_id] = answer["answer"]

    # --------------------------------
    # Create training examples
    # --------------------------------

    examples = []

    for question in questions:

        question_id = question["id"]
        image_id = question["img_id"]
        question_text = question["question"]

        answer_ids = question["answers_ids"]

        # Get the first answer
        if len(answer_ids) == 0:
            continue

        answer_id = answer_ids[0]

        if answer_id not in answer_lookup:
            print(f"Warning: answer ID {answer_id} not found")
            continue

        answer_text = answer_lookup[answer_id]

        # --------------------------------
        # Image filename
        # --------------------------------

        image_filename = f"{image_id}.tif"
        image_path = IMAGE_DIR / image_filename

        # Check image exists
        if not image_path.exists():
            print(f"Warning: image not found: {image_path}")
            continue

        # --------------------------------
        # Create VLM example
        # --------------------------------

        example = {
            "image": f"Images_LR/{image_filename}",
            "question": question_text,
            "answer": answer_text
        }

        examples.append(example)

    # --------------------------------
    # Save JSONL
    # --------------------------------

    output_file = OUTPUT_DIR / f"{split_name}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:

        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"{split_name} examples created: {len(examples)}")
    print(f"Saved to: {output_file}")


# ==============================
# TRAIN
# ==============================

prepare_split(
    "train",
    DATASET_DIR / "TRAIN" / "train_questions.json",
    DATASET_DIR / "TRAIN" / "train_answers.json"
)


# ==============================
# VALIDATION
# ==============================

prepare_split(
    "validation",
    DATASET_DIR / "VALIDATION" / "val_questions.json",
    DATASET_DIR / "VALIDATION" / "val_answers.json"
)


# ==============================
# TEST
# ==============================

prepare_split(
    "test",
    DATASET_DIR / "TEST" / "test_questions.json",
    DATASET_DIR / "TEST" / "test_answers.json"
)


print("\n================================")
print("RSVQA preparation completed!")
print("================================")