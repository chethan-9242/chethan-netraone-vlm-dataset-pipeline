
import json
import re
import gc
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path("/content/netra-one-vlm-pipeline")

IMAGE_DIR = PROJECT_DIR / "data" / "filtered"
OUTPUT_DIR = PROJECT_DIR / "data" / "annotations"

OUTPUT_FILE = OUTPUT_DIR / "vlm_annotations.jsonl"
ERROR_FILE = OUTPUT_DIR / "vlm_errors.jsonl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

print("=" * 60)
print("NETRA-ONE VLM DATASET GENERATION")
print("=" * 60)

print("\nLoading model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

print("Model loaded.")
print("Device:", model.device)


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
Analyze THIS CCTV image carefully.

Use only information that is clearly visible in THIS image.
Do not copy objects, questions, answers, or relationships from any other image.

Do NOT invent:
- object identities
- people
- actions
- movement
- intentions
- locations
- events
- security incidents

This is a still image. Do not say moving, walking, driving, running,
crossing, etc. unless the action is clearly visible.

If a person is riding a motorcycle or scooter, describe them as a rider
or person on a motorcycle/scooter. Do not automatically call every person
a pedestrian.

Use simple, factual descriptions.

Return EXACTLY this JSON structure:

{
  "scene_description": "...",
  "tactical_assessment": "...",
  "vqa": [
    {
      "question_type": "identification",
      "question": "...",
      "answer": "..."
    },
    {
      "question_type": "spatial_reasoning",
      "question": "...",
      "answer": "..."
    },
    {
      "question_type": "threat_evaluation",
      "question": "What is the observable security threat level?",
      "answer": "LOW. ..."
    }
  ]
}

VQA REQUIREMENTS:

1. IDENTIFICATION

Create ONE question about something actually visible in THIS image.

Examples:
- What color is the truck?
- What type of vehicle is near the center?
- What object is visible beside the road?

The answer must directly answer the question.

Do not always use a truck or three-wheeled vehicle.
Choose the object based on THIS image.

2. SPATIAL REASONING

Create ONE question about the relative position of TWO visible objects
in THIS image.

Examples:
- Where is the truck relative to the bus?
- Where is the motorcycle relative to the car?
- What object is beside the truck?

Use only a relationship that is visibly supported.

The answer must directly answer the question.

3. THREAT EVALUATION

The question MUST be exactly:

"What is the observable security threat level?"

Use:

LOW = no clearly visible security threat.

MEDIUM = a clearly visible unusual or potentially concerning condition,
but no immediate serious threat.

HIGH = clear visible evidence of an immediate or serious threat.

Normal traffic, multiple vehicles, pedestrians, congestion, vehicle
types, or a busy road are NOT by themselves security threats.

Do not invent hypothetical risks.

The answer MUST begin with exactly one of:

LOW.
MEDIUM.
HIGH.

Then give a short visible justification.

IMPORTANT:
The VQA questions and answers must be based on THIS image.
Do not reuse a fixed question-answer pair.

Return ONLY valid JSON.
Do not return Markdown.
"""


# ============================================================
# JSON CLEANING
# ============================================================

def clean_json_output(text):

    text = text.strip()

    # Remove markdown fences if the model adds them.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # Find JSON object if extra text was generated.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return text.strip()


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_annotation(data):

    result = {
        "scene_description": str(
            data.get("scene_description", "")
        ).strip(),

        "tactical_assessment": str(
            data.get("tactical_assessment", "")
        ).strip(),

        "vqa": []
    }

    raw_vqa = data.get("vqa", [])

    if isinstance(raw_vqa, list):

        for item in raw_vqa:

            if not isinstance(item, dict):
                continue

            result["vqa"].append({
                "question_type": str(
                    item.get("question_type", "")
                ).strip(),

                "question": str(
                    item.get("question", "")
                ).strip(),

                "answer": str(
                    item.get("answer", "")
                ).strip()
            })

    # Normalize threat answer capitalization.
    for item in result["vqa"]:

        if item["question_type"] == "threat_evaluation":

            answer = item["answer"].strip()

            match = re.match(
                r"^(LOW|MEDIUM|HIGH)\b",
                answer,
                flags=re.IGNORECASE
            )

            if match:

                level = match.group(1).upper()

                remainder = answer[match.end():].strip()

                # Remove repeated punctuation/level.
                remainder = re.sub(
                    r"^[\s\.:;-]+",
                    "",
                    remainder
                )

                remainder = re.sub(
                    r"^(LOW|MEDIUM|HIGH)\b[\s\.:;-]*",
                    "",
                    remainder,
                    flags=re.IGNORECASE
                )

                if remainder:
                    item["answer"] = f"{level}. {remainder}"
                else:
                    item["answer"] = f"{level}."


    return result


# ============================================================
# VALIDATION
# ============================================================

REQUIRED_TYPES = {
    "identification",
    "spatial_reasoning",
    "threat_evaluation"
}


def validate_annotation(data):

    if not isinstance(data, dict):
        return False, "Output is not a JSON object"

    if not isinstance(data.get("scene_description"), str):
        return False, "Invalid scene_description"

    if not data["scene_description"].strip():
        return False, "Empty scene_description"

    if not isinstance(data.get("tactical_assessment"), str):
        return False, "Invalid tactical_assessment"

    if not data["tactical_assessment"].strip():
        return False, "Empty tactical_assessment"

    vqa = data.get("vqa")

    if not isinstance(vqa, list):
        return False, "vqa is not a list"

    if len(vqa) != 3:
        return False, f"Expected 3 VQA pairs, got {len(vqa)}"

    found_types = set()

    for item in vqa:

        if not isinstance(item, dict):
            return False, "VQA item is not an object"

        qtype = item.get("question_type", "")

        if qtype not in REQUIRED_TYPES:
            return False, f"Invalid question type: {qtype}"

        if qtype in found_types:
            return False, f"Duplicate question type: {qtype}"

        found_types.add(qtype)

        question = item.get("question", "")
        answer = item.get("answer", "")

        if not isinstance(question, str) or not question.strip():
            return False, "Empty VQA question"

        if not isinstance(answer, str) or not answer.strip():
            return False, "Empty VQA answer"

        # Reject obvious prompt-template leakage.
        bad_phrases = [
            "actual identification question",
            "actual spatial reasoning question",
            "identification question",
            "spatial reasoning question",
            "what is the actual question",
            "fill in the question"
        ]

        q_lower = question.lower()

        for phrase in bad_phrases:
            if phrase in q_lower:
                return False, "Template question detected"

    if found_types != REQUIRED_TYPES:
        return False, "Missing required VQA type"

    # --------------------------------------------------------
    # Threat validation
    # --------------------------------------------------------

    threat = None

    for item in vqa:
        if item["question_type"] == "threat_evaluation":
            threat = item
            break

    if threat is None:
        return False, "Missing threat evaluation"

    expected_question = "What is the observable security threat level?"

    if threat["question"].strip() != expected_question:
        return False, "Incorrect threat question"

    answer = threat["answer"].strip()

    if not re.match(
        r"^(LOW|MEDIUM|HIGH)\.",
        answer,
        flags=re.IGNORECASE
    ):
        return False, "Threat answer does not start with LOW., MEDIUM., or HIGH."

    # Require something after the threat level.
    remainder = re.sub(
        r"^(LOW|MEDIUM|HIGH)\.\s*",
        "",
        answer,
        flags=re.IGNORECASE
    ).strip()

    if not remainder:
        return False, "Threat answer has no justification"

    return True, "OK"


# ============================================================
# GENERATION
# ============================================================

def generate_annotation(image_path):

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": PROMPT
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False
        )

    # Only keep newly generated tokens.
    input_length = inputs["input_ids"].shape[1]

    generated_ids_trimmed = [
        output_ids[input_length:]
        for output_ids in generated_ids
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0].strip()

    del inputs
    del generated_ids
    del generated_ids_trimmed
    del image
    del image_inputs
    del video_inputs

    return output_text


# ============================================================
# LOAD EXISTING SUCCESSFUL RECORDS
# ============================================================

completed = {}

if OUTPUT_FILE.exists():

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            try:

                record = json.loads(line)

                image_name = record.get("image")

                if image_name:
                    completed[image_name] = record

            except Exception:
                pass


# ============================================================
# IMAGE LIST
# ============================================================

image_files = sorted(
    IMAGE_DIR.glob("*.png")
)

print()
print("Total filtered images:", len(image_files))
print("Already completed:", len(completed))
print()


# ============================================================
# BATCH PROCESSING
# ============================================================

success_count = len(completed)
error_count = 0

for index, image_path in enumerate(
    image_files,
    start=1
):

    image_name = image_path.name

    if image_name in completed:

        print(
            f"[{index}/{len(image_files)}] "
            f"{image_name} -> SKIP"
        )

        continue

    print(
        f"[{index}/{len(image_files)}] "
        f"{image_name}"
    )

    try:

        raw_output = generate_annotation(
            image_path
        )

        cleaned = clean_json_output(
            raw_output
        )

        data = json.loads(cleaned)

        # Normalize BEFORE validation.
        data = normalize_annotation(data)

        valid, message = validate_annotation(data)

        if not valid:

            raise ValueError(message)

        record = {
            "image": image_name,
            "model": MODEL_NAME,
            "annotation": data
        }

        with open(
            OUTPUT_FILE,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

        success_count += 1

        print(
            "  STATUS: ACCEPTED"
        )

    except Exception as e:

        error_count += 1

        error_record = {
            "image": image_name,
            "error": str(e),
            "raw_output": locals().get(
                "raw_output",
                ""
            )
        }

        with open(
            ERROR_FILE,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(
                    error_record,
                    ensure_ascii=False
                )
                + "\n"
            )

        print(
            "  STATUS: REJECTED"
        )

        print(
            "  REASON:",
            str(e)
        )

    finally:

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("PART B COMPLETE")
print("=" * 60)

print("Total images:", len(image_files))
print("Successful:", success_count)
print("Rejected:", error_count)

print()
print("Annotations:")
print(OUTPUT_FILE)

print()
print("Errors:")
print(ERROR_FILE)
