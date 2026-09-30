
# Netra-One Multimodal AI

## VLM Dataset Curation & Auto-Labeling Pipeline

This project implements an automated data curation and auto-labeling pipeline for creating high-quality multimodal Vision-Language Model (VLM) training data from surveillance/security imagery.

The pipeline performs four main stages:

1. Image collection and quality filtering
2. VLM-based scene description and VQA generation
3. Text cleaning and CLIP-based verification
4. LLaVA-style dataset export and dataset statistics

---

## Pipeline Architecture

```text
Surveillance Images
        |
        v
+-------------------------+
| collect_and_filter.py   |
|                         |
| - Dark image filtering  |
| - Blur detection        |
| - Low-information       |
| - pHash deduplication   |
+-------------------------+
        |
        v
Filtered Images
        |
        v
+-------------------------+
| generate_vlm_dataset.py |
|                         |
| - Scene description     |
| - Tactical assessment   |
| - Identification VQA    |
| - Spatial reasoning VQA |
| - Threat evaluation VQA |
+-------------------------+
        |
        v
VLM Annotations
        |
        v
+-------------------------+
| clean_and_verify.py     |
|                         |
| - Text normalization    |
| - JSON validation       |
| - CLIP similarity       |
| - Low-score quarantine  |
+-------------------------+
        |
        v
Verified Dataset
        |
        v
+-------------------------+
| export_dataset.py       |
|                         |
| - LLaVA JSONL           |
| - Sample JSONL          |
| - Dataset statistics    |
+-------------------------+
        |
        v
Final VLM Dataset
````

---

## Dataset Source

The pipeline was evaluated using the validation split of the **UVH-26** dataset from the Indian Institute of Science.

Dataset:

[https://huggingface.co/datasets/iisc-aim/UVH-26](https://huggingface.co/datasets/iisc-aim/UVH-26)

The validation split contains **Bengaluru Safe City CCTV imagery**.

For this project:

* 150 candidate images were selected
* 150 images were successfully downloaded
* 148 images passed the image-quality filtering stage

The source imagery is surveillance/CCTV imagery and is not a military-specific dataset.

---

# Stage 1 — Image Collection and Filtering

### Script

```text
collect_and_filter.py
```

The first stage prepares the raw images for VLM annotation.

The pipeline performs:

* Image validity checking
* Dark image detection
* Motion-blur detection using Laplacian variance
* Low-information detection using edge density
* Near-duplicate detection using perceptual hashing (pHash)

### Filtering Rules

The pipeline uses image-quality measurements including:

* Brightness
* Variance of Laplacian
* Edge density
* Perceptual hash distance

The final configuration uses:

```text
Darkness threshold: 30
Blur threshold: 50
Degraded blur threshold: 70
Degraded edge threshold: 0.012
Low-information edge threshold: 0.008
pHash distance threshold: 6
```

### Stage 1 Results

```text
Input images:              150
Valid images:              150
Quality-passed images:     148
Near-duplicates removed:    0
Final filtered images:     148
```

The rejected images are preserved separately in the quarantine directory.

---

# Stage 2 — VLM Auto-Labeling

### Script

```text
generate_vlm_dataset.py
```

The second stage uses an open Vision-Language Model to automatically generate annotations for the filtered images.

### Model

```text
Qwen/Qwen2-VL-2B-Instruct
```

The model generates:

### 1. Scene Description

A description of the visible scene and objects.

### 2. Tactical Security Assessment

A security-oriented assessment based only on observable information in the image.

### 3. Identification VQA

An image-specific question and answer involving visible objects.

### 4. Spatial Reasoning VQA

An image-specific question and answer describing spatial relationships.

### 5. Threat Evaluation VQA

A question and answer containing:

```text
LOW
MEDIUM
HIGH
```

with a justification based on visible evidence.

The prompt instructs the model not to invent people, objects, movement, locations, actions, or security incidents that are not visibly supported by the image.

### Stage 2 Results

```text
Filtered images processed: 148
Successful annotations:    148
VQA pairs per image:         3
```

Each successful annotation contains:

```text
scene_description
tactical_assessment
vqa
```

The VQA section contains:

```text
identification
spatial_reasoning
threat_evaluation
```

---

# Stage 3 — Cleaning and CLIP Verification

### Script

```text
clean_and_verify.py
```

The third stage verifies the generated annotations.

The pipeline performs:

* Text normalization
* Removal of formatting artifacts
* JSON structure validation
* VQA structure validation
* Threat-answer validation
* Image-text similarity calculation using CLIP

### CLIP Model

```text
openai/clip-vit-base-patch32
```

The CLIP score is calculated using cosine similarity between image and text embeddings.

### CLIP Threshold

```text
0.24
```

Records below this threshold are flagged and moved to a separate quarantine directory rather than being silently deleted.

### Stage 3 Results

```text
Annotation records checked: 148
CLIP records checked:       148
Validation failures:          0
CLIP-flagged records:        14
Clean verified records:     134
Quarantined records:         14
```

### CLIP Score Statistics

For the complete 148-record annotation set:

```text
Minimum CLIP score:  0.2015
Maximum CLIP score:  0.3457
Average CLIP score:  0.2727
Threshold:           0.2400
```

The 14 records below the threshold are preserved in:

```text
data/quarantine_clip/
```

The final verified dataset contains 134 records.

---

# Stage 4 — Dataset Export

### Script

```text
export_dataset.py
```

The final stage converts the verified annotations into a LLaVA-style multimodal JSONL dataset.

Each record contains information such as:

```text
id
image
conversations
metadata
```

The pipeline also generates a smaller sample dataset for inspection and submission.

### Stage 4 Results

```text
Final verified records:       134
Full exported records:        134
Sample dataset records:        75
Sample images:                 75
```

The final dataset is available at:

```text
data/exported/netra_one_llava_dataset.jsonl
```

The sample dataset is:

```text
sample_dataset.jsonl
```

---

# Dataset Statistics

The generated `dataset_stats.json` contains statistics about the final verified dataset.

### Final Dataset

```text
Total tokens:             15,355
Unique tokens:               463
Vocabulary diversity:      0.0302

Minimum record length:       77
Maximum record length:      152
Mean record length:       114.59
```

### CLIP Distribution in Final Dataset

```text
Minimum CLIP score:  0.2402
Maximum CLIP score:  0.3457
Average CLIP score:  0.2771
```

---

# Project Structure

```text
chethan-netraone-vlm-dataset-pipeline/
│
├── README.md
├── requirements.txt
│
├── collect_and_filter.py
├── generate_vlm_dataset.py
├── clean_and_verify.py
├── export_dataset.py
│
├── sample_dataset.jsonl
├── dataset_stats.json
│
├── sample_images/
│
└── data/
    │
    ├── raw/
    ├── filtered/
    ├── quarantine/
    ├── quarantine_clip/
    │
    ├── annotations/
    │   ├── vlm_annotations.jsonl
    │   ├── vlm_errors.jsonl
    │   ├── clip_scores.jsonl
    │   └── cleaning_report.json
    │
    ├── cleaned/
    │   └── verified_annotations.jsonl
    │
    └── exported/
        └── netra_one_llava_dataset.jsonl
```

---

# Main Output Files

| File                                          | Description                                |
| --------------------------------------------- | ------------------------------------------ |
| `data/filter_report.json`                     | Image filtering statistics                 |
| `data/annotations/vlm_annotations.jsonl`      | VLM-generated annotations                  |
| `data/annotations/clip_scores.jsonl`          | CLIP similarity scores                     |
| `data/annotations/cleaning_report.json`       | Cleaning and verification statistics       |
| `data/cleaned/verified_annotations.jsonl`     | Final verified annotations                 |
| `data/exported/netra_one_llava_dataset.jsonl` | Final LLaVA-style dataset                  |
| `sample_dataset.jsonl`                        | 75-sample dataset                          |
| `dataset_stats.json`                          | Dataset statistics                         |
| `sample_images/`                              | Images corresponding to the sample dataset |

---

# Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

The VLM and CLIP stages require a CUDA-compatible GPU for practical execution.

---

# Running the Pipeline

Run the four stages in order:

```bash
python collect_and_filter.py
```

```bash
python generate_vlm_dataset.py
```

```bash
python clean_and_verify.py
```

```bash
python export_dataset.py
```

The VLM generation stage downloads and uses the Qwen2-VL-2B-Instruct model.

---

# Quality Control Strategy

The pipeline uses multiple quality-control stages instead of relying only on VLM-generated text.

```text
Image Quality
     |
     v
Dark / Blur / Information Filtering
     |
     v
pHash Deduplication
     |
     v
VLM Annotation
     |
     v
JSON + Text Validation
     |
     v
CLIP Image-Text Similarity
     |
     v
Low-score Quarantine
     |
     v
Verified Dataset
```

This allows low-quality images and potentially weak image-text pairs to be separated before final dataset export.

---

# Limitations

* The source imagery is Bengaluru Safe City CCTV imagery and is not a military-specific dataset.
* Threat assessments are based only on visually observable information in individual frames.
* The VLM generates annotations automatically, so annotations may still require human review for production use.
* CLIP similarity is used as a quality signal and does not guarantee complete semantic correctness.
* The threat classification is based on the information visible in a single image and should not be interpreted as a real-world security decision.
* The dataset uses automatically generated descriptions and VQA pairs, so further human review can improve annotation quality.

---

## Important: Accessing the Project After Download

When downloading the repository as a ZIP from GitHub, the extracted folder may contain another folder with the same name:

```text
chethan-netraone-vlm-dataset-pipeline-main/
└── chethan-netraone-vlm-dataset-pipeline-main/
    ├── README.md
    ├── requirements.txt
    ├── collect_and_filter.py
    ├── generate_vlm_dataset.py
    ├── clean_and_verify.py
    ├── export_dataset.py
    ├── sample_dataset.jsonl
    ├── sample_images/
    └── data/


# Summary

The completed pipeline provides an end-to-end workflow for converting raw surveillance imagery into a filtered, automatically annotated, CLIP-verified, and LLaVA-compatible multimodal dataset.

Final pipeline results:

```text
150 raw images
      |
      v
148 filtered images
      |
      v
148 VLM annotations
      |
      v
14 CLIP-flagged records
      |
      v
134 verified records
      |
      v
134 final exported records
      |
      v
75-record sample dataset
