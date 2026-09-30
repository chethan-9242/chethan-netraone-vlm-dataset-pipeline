
import os
import json
import shutil
import cv2
import imagehash
from PIL import Image
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

BASE_DIR = "/content/netra-one-vlm-pipeline"

RAW_DIR = os.path.join(
    BASE_DIR, "data", "raw"
)

FILTERED_DIR = os.path.join(
    BASE_DIR, "data", "filtered"
)

QUARANTINE_DIR = os.path.join(
    BASE_DIR, "data", "quarantine"
)

REPORT_FILE = os.path.join(
    BASE_DIR, "data", "filter_report.json"
)


# ============================================================
# FILTERING PARAMETERS
# ============================================================

DARK_THRESHOLD = 30

BLUR_THRESHOLD = 50

LOW_INFORMATION_THRESHOLD = 0.008

DEGRADED_BLUR_THRESHOLD = 70

DEGRADED_EDGE_THRESHOLD = 0.012

PHASH_THRESHOLD = 6


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(FILTERED_DIR, exist_ok=True)
os.makedirs(QUARANTINE_DIR, exist_ok=True)


# ============================================================
# IMAGE QUALITY MEASUREMENT
# ============================================================

def calculate_metrics(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    brightness = float(gray.mean())

    blur_score = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = float(
        (edges > 0).mean()
    )

    return brightness, blur_score, edge_density


# ============================================================
# QUALITY DECISION
# ============================================================

def is_quality_good(
    brightness,
    blur_score,
    edge_density
):

    # Completely / extremely dark
    if brightness < DARK_THRESHOLD:
        return False, "dark"

    # Strong blur
    if blur_score < BLUR_THRESHOLD:
        return False, "blur"

    # Very low information
    if edge_density < LOW_INFORMATION_THRESHOLD:
        return False, "low_information"

    # Combination of moderate blur and very low edge content
    if (
        blur_score < DEGRADED_BLUR_THRESHOLD
        and
        edge_density < DEGRADED_EDGE_THRESHOLD
    ):
        return False, "degraded_quality"

    return True, "passed"


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("NETRA-ONE PART A: IMAGE COLLECTION & FILTERING")
print("=" * 60)


files = sorted([
    f
    for f in os.listdir(RAW_DIR)
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    )
])


print(f"Input images: {len(files)}")


valid_count = 0
quality_passed = 0
quality_rejected = 0
duplicate_count = 0

rejection_reasons = {}

accepted_hashes = []


for filename in tqdm(
    files,
    desc="Filtering images"
):

    source_path = os.path.join(
        RAW_DIR,
        filename
    )

    image = cv2.imread(
        source_path
    )

    # --------------------------------------------------------
    # Invalid image
    # --------------------------------------------------------

    if image is None:

        quality_rejected += 1

        rejection_reasons.setdefault(
            "invalid_image",
            []
        ).append(filename)

        destination = os.path.join(
            QUARANTINE_DIR,
            filename
        )

        shutil.copy2(
            source_path,
            destination
        )

        continue


    valid_count += 1


    # --------------------------------------------------------
    # Quality metrics
    # --------------------------------------------------------

    brightness, blur_score, edge_density = calculate_metrics(
        image
    )


    passed, reason = is_quality_good(
        brightness,
        blur_score,
        edge_density
    )


    if not passed:

        quality_rejected += 1

        rejection_reasons.setdefault(
            reason,
            []
        ).append(filename)

        destination = os.path.join(
            QUARANTINE_DIR,
            filename
        )

        shutil.copy2(
            source_path,
            destination
        )

        continue


    # --------------------------------------------------------
    # Perceptual hash
    # --------------------------------------------------------

    try:

        pil_image = Image.open(
            source_path
        )

        current_hash = imagehash.phash(
            pil_image
        )

    except Exception:

        quality_rejected += 1

        rejection_reasons.setdefault(
            "hash_error",
            []
        ).append(filename)

        destination = os.path.join(
            QUARANTINE_DIR,
            filename
        )

        shutil.copy2(
            source_path,
            destination
        )

        continue


    # --------------------------------------------------------
    # Near-duplicate detection
    # --------------------------------------------------------

    is_duplicate = False

    for previous_hash in accepted_hashes:

        distance = current_hash - previous_hash

        if distance <= PHASH_THRESHOLD:

            is_duplicate = True
            break


    if is_duplicate:

        duplicate_count += 1

        rejection_reasons.setdefault(
            "near_duplicate",
            []
        ).append(filename)

        destination = os.path.join(
            QUARANTINE_DIR,
            filename
        )

        shutil.copy2(
            source_path,
            destination
        )

        continue


    # --------------------------------------------------------
    # Accept image
    # --------------------------------------------------------

    accepted_hashes.append(
        current_hash
    )

    quality_passed += 1

    destination = os.path.join(
        FILTERED_DIR,
        filename
    )

    shutil.copy2(
        source_path,
        destination
    )


# ============================================================
# REPORT
# ============================================================

final_images = len([
    f
    for f in os.listdir(FILTERED_DIR)
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    )
])


report = {

    "pipeline": "Netra-One VLM Dataset Curation",

    "stage": "Part A - Image Collection and Filtering",

    "input_images": len(files),

    "valid_images": valid_count,

    "quality_passed": quality_passed,

    "quality_rejected": quality_rejected,

    "near_duplicates_removed": duplicate_count,

    "final_images": final_images,

    "thresholds": {

        "dark_threshold": DARK_THRESHOLD,

        "blur_threshold": BLUR_THRESHOLD,

        "low_information_threshold":
            LOW_INFORMATION_THRESHOLD,

        "degraded_blur_threshold":
            DEGRADED_BLUR_THRESHOLD,

        "degraded_edge_threshold":
            DEGRADED_EDGE_THRESHOLD,

        "phash_threshold":
            PHASH_THRESHOLD
    },

    "rejection_reasons": rejection_reasons
}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("PART A COMPLETE")
print("=" * 60)

print(f"Input images:             {len(files)}")
print(f"Valid images:             {valid_count}")
print(f"Quality passed:           {quality_passed}")
print(f"Quality rejected:         {quality_rejected}")
print(f"Near duplicates removed:  {duplicate_count}")
print(f"Final images:              {final_images}")

print()
print("Report:")
print(REPORT_FILE)
