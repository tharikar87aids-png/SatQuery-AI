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

def prepare_split(split_name, question_file, answer_file, image_file):

    print(f"\nPreparing {split_name}...")

    # Load files
    questions_data = load_json(question_file)
    answers_data = load_json(answer_file)
    images_data = load_json(image_file)

    # Get lists
    questions = questions_data["questions"]
    answers = answers_data["answers"]
    images = images_data["images"]

    print(f"Questions found: {len(questions)}")
    print(f"Answers found: {len(answers)}")
    print(f"Images found: {len(images)}")


    # ==============================
    # CREATE ANSWER LOOKUP
    # ==============================

    answer_lookup = {}

    for answer in answers:

        answer_id = answer.get("id")
        answer_text = answer.get("answer")

        if answer_id is None or answer_text is None:
            print("Skipping invalid answer record:", answer)
            continue

        answer_lookup[answer_id] = answer_text


    # ==============================
    # CREATE QUESTION → IMAGE MAPPING
    # ==============================

    question_to_image = {}

    for image in images:

        image_id = image["id"]

        for question_id in image["questions_ids"]:

            question_to_image[question_id] = image_id


    print(f"Question-image mappings: {len(question_to_image)}")


    # ==============================
    # CREATE VLM EXAMPLES
    # ==============================

    examples = []

    for question in questions:

        question_id = question["id"]
        question_text = question["question"]

        # ------------------------------
        # Find image using question ID
        # ------------------------------

        if question_id not in question_to_image:

            print(
                f"Warning: image mapping not found "
                f"for question {question_id}"
            )

            continue

        image_id = question_to_image[question_id]


        # ------------------------------
        # Get answer ID
        # ------------------------------

        answer_ids = question["answers_ids"]

        if len(answer_ids) == 0:
            print(
                f"Warning: no answer found "
                f"for question {question_id}"
            )

            continue

        answer_id = answer_ids[0]


        # ------------------------------
        # Find actual answer
        # ------------------------------

        if answer_id not in answer_lookup:

            print(
                f"Warning: answer ID {answer_id} "
                f"not found"
            )

            continue

        answer_text = answer_lookup[answer_id]


        # ------------------------------
        # Image filename
        # ------------------------------

        image_filename = f"{image_id}.tif"

        image_path = IMAGE_DIR / image_filename


        # ------------------------------
        # Check image exists
        # ------------------------------

        if not image_path.exists():

            print(
                f"Warning: image not found: "
                f"{image_path}"
            )

            continue


        # ==============================
        # CREATE VLM EXAMPLE
        # ==============================

        example = {

            "image": f"Images_LR/{image_filename}",

            "question": question_text,

            "answer": answer_text

        }

        examples.append(example)


    # ==============================
    # SAVE JSONL
    # ==============================

    output_file = OUTPUT_DIR / f"{split_name}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:

        for example in examples:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                + "\n"
            )


    print(
        f"{split_name} examples created: "
        f"{len(examples)}"
    )

    print(
        f"Saved to: {output_file}"
    )


# ==============================
# TRAIN
# ==============================

prepare_split(
    "train",

    DATASET_DIR / "TRAIN" / "train_questions.json",

    DATASET_DIR / "TRAIN" / "train_answers.json",

    DATASET_DIR / "TRAIN" / "train_images.json"
)


# ==============================
# VALIDATION
# ==============================

prepare_split(
    "validation",

    DATASET_DIR / "VALIDATION" / "val_questions.json",

    DATASET_DIR / "VALIDATION" / "val_answers.json",

    DATASET_DIR / "VALIDATION" / "val_images.json"
)


# ==============================
# TEST
# ==============================

prepare_split(
    "test",

    DATASET_DIR / "TEST" / "test_questions.json",

    DATASET_DIR / "TEST" / "test_answers.json",

    DATASET_DIR / "TEST" / "test_images.json"
)


# ==============================
# FINISHED
# ==============================

print("\n================================")
print("RSVQA preparation completed!")
print("================================")