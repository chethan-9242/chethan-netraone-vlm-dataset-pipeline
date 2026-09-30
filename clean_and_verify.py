
import os
import json
import re
import shutil


# ============================================================
# PATHS
# ============================================================

BASE_DIR = "/content/netra-one-vlm-pipeline"

ANNOTATIONS_FILE = os.path.join(
    BASE_DIR, "data", "annotations", "vlm_annotations.jsonl"
)

CLIP_FILE = os.path.join(
    BASE_DIR, "data", "annotations", "clip_scores.jsonl"
)

CLEANED_DIR = os.path.join(
    BASE_DIR, "data", "cleaned"
)

QUARANTINE_DIR = os.path.join(
    BASE_DIR, "data", "quarantine_clip"
)

CLEANED_FILE = os.path.join(
    CLEANED_DIR, "verified_annotations.jsonl"
)

REPORT_FILE = os.path.join(
    BASE_DIR, "data", "annotations", "cleaning_report.json"
)

CLIP_THRESHOLD = 0.24


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(QUARANTINE_DIR, exist_ok=True)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Clean generated text conservatively without changing
    its actual meaning.
    """

    if not isinstance(text, str):
        return ""

    text = text.strip()

    # Remove accidental Markdown code fences
    text = re.sub(r"```(?:json|text)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")

    # Remove common model-control tokens
    text = re.sub(r"<\|[^>]*\|>", "", text)
    text = text.replace("[INST]", "")
    text = text.replace("[/INST]", "")

    # Remove common assistant prefixes only at the beginning
    text = re.sub(
        r"^(assistant|answer|response)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove generic opening boilerplate
    text = re.sub(
        r"^(sure|certainly|here is the answer|here's the answer)\s*[:,]?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# VQA VALIDATION
# ============================================================

EXPECTED_TYPES = {
    "identification",
    "spatial_reasoning",
    "threat_evaluation"
}


def validate_annotation(annotation):
    """
    Validate the final annotation structure.
    """

    if not isinstance(annotation, dict):
        return False, "Annotation is not a JSON object"

    required_fields = [
        "scene_description",
        "tactical_assessment",
        "vqa"
    ]

    for field in required_fields:
        if field not in annotation:
            return False, f"Missing field: {field}"

    if not isinstance(annotation["scene_description"], str):
        return False, "scene_description is not text"

    if not isinstance(annotation["tactical_assessment"], str):
        return False, "tactical_assessment is not text"

    if not annotation["scene_description"].strip():
        return False, "Empty scene_description"

    if not annotation["tactical_assessment"].strip():
        return False, "Empty tactical_assessment"

    vqa = annotation["vqa"]

    if not isinstance(vqa, list):
        return False, "VQA is not a list"

    if len(vqa) != 3:
        return False, f"Expected 3 VQA pairs, found {len(vqa)}"

    found_types = set()

    for item in vqa:

        if not isinstance(item, dict):
            return False, "VQA item is not an object"

        for field in ["question_type", "question", "answer"]:
            if field not in item:
                return False, f"VQA missing field: {field}"

            if not isinstance(item[field], str):
                return False, f"VQA field {field} is not text"

            if not item[field].strip():
                return False, f"Empty VQA field: {field}"

        question_type = item["question_type"].strip()

        if question_type not in EXPECTED_TYPES:
            return False, f"Unexpected VQA type: {question_type}"

        if question_type in found_types:
            return False, f"Duplicate VQA type: {question_type}"

        found_types.add(question_type)

        # Reject obvious template questions
        bad_question_patterns = [
            "actual identification question",
            "identification question",
            "spatial reasoning question",
            "threat evaluation question"
        ]

        question_lower = item["question"].lower()

        for pattern in bad_question_patterns:
            if pattern in question_lower:
                return False, f"Template question detected: {item['question']}"

    if found_types != EXPECTED_TYPES:
        return False, "Required VQA types are incomplete"

    # Threat evaluation must contain LOW / MEDIUM / HIGH
    threat_item = next(
        item for item in vqa
        if item["question_type"] == "threat_evaluation"
    )

    threat_question = threat_item["question"].strip().lower()
    threat_answer = threat_item["answer"].strip()

    if "threat" not in threat_question:
        return False, "Threat VQA question does not mention threat"

    if not re.match(
        r"^(LOW|MEDIUM|HIGH)\b",
        threat_answer,
        flags=re.IGNORECASE
    ):
        return False, "Threat answer does not start with LOW/MEDIUM/HIGH"

    # A threat answer may contain only the threat level.
    # The cleaning stage will add the existing tactical
    # assessment as the justification when necessary.
    if len(threat_answer.split()) < 3:
        return True, "Threat level accepted; justification will be normalized"

    return True, "OK"


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    records = []

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"Warning: invalid JSON at line {line_number}: {e}"
                )

    return records


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("NETRA-ONE PART C: CLEANING & VERIFICATION")
print("=" * 60)

annotations = load_jsonl(ANNOTATIONS_FILE)
clip_records = load_jsonl(CLIP_FILE)

print(f"Annotation records: {len(annotations)}")
print(f"CLIP records:       {len(clip_records)}")


# Create CLIP lookup
clip_lookup = {}

for record in clip_records:

    image_name = record.get("image")

    if image_name is None:
        continue

    clip_score = record.get("clip_score")

    if isinstance(clip_score, (int, float)):
        clip_lookup[image_name] = float(clip_score)


# ============================================================
# PROCESS
# ============================================================

clean_records = []

quarantined = []

invalid_records = []

processed = 0
clip_flagged = 0
validation_failed = 0


for record in annotations:

    processed += 1

    image_name = record.get("image")

    annotation = record.get("annotation")

    if not image_name:
        validation_failed += 1

        invalid_records.append({
            "image": image_name,
            "reason": "Missing image name"
        })

        continue

    if not isinstance(annotation, dict):
        validation_failed += 1

        invalid_records.append({
            "image": image_name,
            "reason": "Annotation is not an object"
        })

        continue

    # --------------------------------------------------------
    # Normalize text
    # --------------------------------------------------------

    annotation["scene_description"] = normalize_text(
        annotation.get("scene_description", "")
    )

    annotation["tactical_assessment"] = normalize_text(
        annotation.get("tactical_assessment", "")
    )

    normalized_vqa = []

    for item in annotation.get("vqa", []):

        if not isinstance(item, dict):
            continue

        normalized_vqa.append({
            "question_type": normalize_text(
                item.get("question_type", "")
            ),
            "question": normalize_text(
                item.get("question", "")
            ),
            "answer": normalize_text(
                item.get("answer", "")
            )
        })

    annotation["vqa"] = normalized_vqa

    # --------------------------------------------------------
    # Validate structure
    # --------------------------------------------------------

    valid, reason = validate_annotation(annotation)

    if not valid:

        validation_failed += 1

        invalid_records.append({
            "image": image_name,
            "reason": reason
        })

        continue

    # --------------------------------------------------------
    # Normalize short threat answers
    # --------------------------------------------------------

    for item in annotation["vqa"]:

        if item["question_type"] != "threat_evaluation":
            continue

        threat_answer = item["answer"].strip()

        # If the model returned only LOW/MEDIUM/HIGH,
        # reuse the justification already present in the
        # tactical assessment instead of inventing new text.
        if len(threat_answer.split()) < 3:

            level_match = re.match(
                r"^(LOW|MEDIUM|HIGH)\\b",
                threat_answer,
                flags=re.IGNORECASE
            )

            if level_match:

                level = level_match.group(1).upper()

                tactical = annotation["tactical_assessment"]

                # Handle dictionary-style tactical output
                if isinstance(tactical, dict):

                    reason = tactical.get("reason", "").strip()

                    if reason:
                        item["answer"] = f"{level}. {reason}"

                else:

                    tactical_text = str(tactical).strip()

                    # Remove an existing leading threat level
                    tactical_text = re.sub(
                        r"^(LOW|MEDIUM|HIGH)\\.?\\s*",
                        "",
                        tactical_text,
                        flags=re.IGNORECASE
                    )

                    if tactical_text:
                        item["answer"] = (
                            f"{level}. {tactical_text}"
                        )

    # --------------------------------------------------------
    # CLIP score
    # --------------------------------------------------------

    clip_score = clip_lookup.get(image_name)

    if clip_score is None:

        validation_failed += 1

        invalid_records.append({
            "image": image_name,
            "reason": "Missing CLIP score"
        })

        continue

    # Add CLIP score to record
    record["clip_score"] = round(clip_score, 6)

    # --------------------------------------------------------
    # Flag low CLIP score
    # --------------------------------------------------------

    if clip_score < CLIP_THRESHOLD:

        clip_flagged += 1

        quarantined.append(record)

        continue

    # --------------------------------------------------------
    # Accepted clean record
    # --------------------------------------------------------

    clean_records.append(record)


# ============================================================
# WRITE CLEAN DATASET
# ============================================================

with open(CLEANED_FILE, "w", encoding="utf-8") as f:

    for record in clean_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


# ============================================================
# QUARANTINE LOW CLIP IMAGES
# ============================================================

filtered_image_dir = os.path.join(
    BASE_DIR, "data", "filtered"
)

for record in quarantined:

    image_name = record["image"]

    source = os.path.join(
        filtered_image_dir,
        image_name
    )

    destination = os.path.join(
        QUARANTINE_DIR,
        image_name
    )

    if os.path.exists(source):
        shutil.copy2(source, destination)


# ============================================================
# REPORT
# ============================================================

report = {
    "pipeline": "Netra-One VLM Dataset Curation",
    "stage": "Part C - Cleaning and Verification",

    "input_annotation_records": len(annotations),

    "clip_records": len(clip_records),

    "json_and_structure_valid": len(annotations) - validation_failed,

    "validation_failed": validation_failed,

    "clip_threshold": CLIP_THRESHOLD,

    "clip_flagged": clip_flagged,

    "clean_verified_records": len(clean_records),

    "quarantined_images": len(quarantined),

    "cleaned_dataset": CLEANED_FILE,

    "quarantine_directory": QUARANTINE_DIR,

    "invalid_records": invalid_records
}


with open(REPORT_FILE, "w", encoding="utf-8") as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("PART C COMPLETE")
print("=" * 60)

print(f"Input records:          {len(annotations)}")
print(f"Validation failures:    {validation_failed}")
print(f"CLIP flagged (<0.24):   {clip_flagged}")
print(f"Clean verified records: {len(clean_records)}")
print(f"Quarantined images:     {len(quarantined)}")

print()
print("Clean dataset:")
print(CLEANED_FILE)

print()
print("Quarantine:")
print(QUARANTINE_DIR)

print()
print("Report:")
print(REPORT_FILE)
