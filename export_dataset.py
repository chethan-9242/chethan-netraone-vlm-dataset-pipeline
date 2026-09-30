
import os
import json
import re
import math
from collections import Counter

BASE_DIR = "/content/netra-one-vlm-pipeline"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "verified_annotations.jsonl"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "exported"
)

FINAL_JSONL = os.path.join(
    OUTPUT_DIR,
    "netra_one_llava_dataset.jsonl"
)

SAMPLE_JSONL = os.path.join(
    BASE_DIR,
    "sample_dataset.jsonl"
)

STATS_FILE = os.path.join(
    BASE_DIR,
    "dataset_stats.json"
)

SAMPLE_SIZE = 75


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD VERIFIED DATA
# ============================================================

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        records.append(json.loads(line))


print("=" * 60)
print("NETRA-ONE PART D: DATASET EXPORT")
print("=" * 60)

print(f"Verified input records: {len(records)}")


# ============================================================
# TOKEN / WORD STATISTICS
# ============================================================

def tokenize_text(text):
    """
    Lightweight tokenization for dataset statistics.
    """

    if not isinstance(text, str):
        return []

    return re.findall(
        r"\b[\w'-]+\b",
        text.lower()
    )


all_tokens = []

sample_lengths = []

all_clip_scores = []


# ============================================================
# CONVERT TO LLAVA-STYLE FORMAT
# ============================================================

exported_records = []


for index, record in enumerate(records, start=1):

    image_name = record["image"]
    annotation = record["annotation"]

    scene = annotation["scene_description"]
    tactical = annotation["tactical_assessment"]

    conversations = []

    # --------------------------------------------------------
    # Scene description
    # --------------------------------------------------------

    conversations.append({
        "from": "human",
        "value": (
            "<image>\n"
            "Describe the visible scene in detail."
        )
    })

    conversations.append({
        "from": "gpt",
        "value": scene
    })

    # --------------------------------------------------------
    # Tactical assessment
    # --------------------------------------------------------

    conversations.append({
        "from": "human",
        "value": (
            "<image>\n"
            "Provide a security assessment based "
            "only on visible evidence."
        )
    })

    conversations.append({
        "from": "gpt",
        "value": tactical
    })

    # --------------------------------------------------------
    # VQA pairs
    # --------------------------------------------------------

    for vqa in annotation["vqa"]:

        conversations.append({
            "from": "human",
            "value": (
                "<image>\n"
                + vqa["question"]
            )
        })

        conversations.append({
            "from": "gpt",
            "value": vqa["answer"]
        })

    exported = {
        "id": f"netra_one_{index:04d}",
        "image": f"sample_images/{image_name}",
        "conversations": conversations,
        "metadata": {
            "source_image": image_name,
            "clip_score": record.get("clip_score")
        }
    }

    exported_records.append(exported)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    record_tokens = []

    for conversation in conversations:

        tokens = tokenize_text(
            conversation["value"]
        )

        all_tokens.extend(tokens)
        record_tokens.extend(tokens)

    sample_lengths.append(
        len(record_tokens)
    )

    clip_score = record.get("clip_score")

    if isinstance(clip_score, (int, float)):
        all_clip_scores.append(
            float(clip_score)
        )


# ============================================================
# WRITE FULL DATASET
# ============================================================

with open(FINAL_JSONL, "w", encoding="utf-8") as f:

    for record in exported_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


# ============================================================
# WRITE SAMPLE DATASET
# ============================================================

sample_records = exported_records[:SAMPLE_SIZE]

with open(SAMPLE_JSONL, "w", encoding="utf-8") as f:

    for record in sample_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


# ============================================================
# TOKEN STATISTICS
# ============================================================

total_tokens = len(all_tokens)

unique_tokens = len(set(all_tokens))

vocabulary_diversity = (
    unique_tokens / total_tokens
    if total_tokens > 0
    else 0
)


if sample_lengths:

    sorted_lengths = sorted(sample_lengths)

    mean_length = (
        sum(sample_lengths) / len(sample_lengths)
    )

    median_length = sorted_lengths[
        len(sorted_lengths) // 2
    ]

    min_length = min(sample_lengths)
    max_length = max(sample_lengths)

else:

    mean_length = 0
    median_length = 0
    min_length = 0
    max_length = 0


# ============================================================
# CLIP DISTRIBUTION
# ============================================================

if all_clip_scores:

    clip_min = min(all_clip_scores)
    clip_max = max(all_clip_scores)

    clip_mean = (
        sum(all_clip_scores) /
        len(all_clip_scores)
    )

    clip_sorted = sorted(all_clip_scores)

    clip_median = clip_sorted[
        len(clip_sorted) // 2
    ]

else:

    clip_min = 0
    clip_max = 0
    clip_mean = 0
    clip_median = 0


# Histogram-style distribution

clip_bins = {
    "0.20-0.24": 0,
    "0.24-0.26": 0,
    "0.26-0.28": 0,
    "0.28-0.30": 0,
    "0.30-0.32": 0,
    "0.32-0.34": 0,
    "0.34+": 0
}


for score in all_clip_scores:

    if score < 0.24:
        clip_bins["0.20-0.24"] += 1

    elif score < 0.26:
        clip_bins["0.24-0.26"] += 1

    elif score < 0.28:
        clip_bins["0.26-0.28"] += 1

    elif score < 0.30:
        clip_bins["0.28-0.30"] += 1

    elif score < 0.32:
        clip_bins["0.30-0.32"] += 1

    elif score < 0.34:
        clip_bins["0.32-0.34"] += 1

    else:
        clip_bins["0.34+"] += 1


# ============================================================
# DATASET STATISTICS
# ============================================================

stats = {

    "dataset": "Netra-One Multimodal VLM Dataset",

    "input_records": len(records),

    "exported_records": len(exported_records),

    "sample_records": len(sample_records),

    "format": "LLaVA-style JSONL",

    "token_statistics": {

        "total_tokens": total_tokens,

        "unique_tokens": unique_tokens,

        "vocabulary_diversity": round(
            vocabulary_diversity,
            4
        ),

        "record_token_length": {

            "minimum": min_length,

            "maximum": max_length,

            "mean": round(
                mean_length,
                2
            ),

            "median": median_length
        }
    },

    "clip_score_distribution": {

        "count": len(all_clip_scores),

        "minimum": round(
            clip_min,
            4
        ),

        "maximum": round(
            clip_max,
            4
        ),

        "mean": round(
            clip_mean,
            4
        ),

        "median": round(
            clip_median,
            4
        ),

        "bins": clip_bins
    },

    "files": {

        "full_dataset": FINAL_JSONL,

        "sample_dataset": SAMPLE_JSONL
    }
}


# ============================================================
# WRITE STATISTICS
# ============================================================

with open(STATS_FILE, "w", encoding="utf-8") as f:

    json.dump(
        stats,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("PART D EXPORT COMPLETE")
print("=" * 60)

print(f"Full exported records: {len(exported_records)}")
print(f"Sample records:        {len(sample_records)}")

print()
print("Token statistics:")
print(f"Total tokens:           {total_tokens}")
print(f"Unique tokens:          {unique_tokens}")
print(f"Vocabulary diversity:   {vocabulary_diversity:.4f}")
print(f"Min record length:      {min_length}")
print(f"Max record length:      {max_length}")
print(f"Mean record length:     {mean_length:.2f}")

print()
print("CLIP statistics:")
print(f"Minimum:                {clip_min:.4f}")
print(f"Maximum:                {clip_max:.4f}")
print(f"Mean:                   {clip_mean:.4f}")

print()
print("Full dataset:")
print(FINAL_JSONL)

print()
print("Sample dataset:")
print(SAMPLE_JSONL)

print()
print("Statistics:")
print(STATS_FILE)
