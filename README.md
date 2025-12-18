[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hgNAtOO3)

# Podcast Emotion and Content Analysis: A Multi-Level NLP Pipeline

## Table of Contents
- [Project Overview](#project-overview)
- [Team Contributions](#team-contributions)
- [The Multi-Task Pipeline](#the-multi-task-pipeline)
- [Architecture & Implementation](#architecture--implementation)
- [Dataset & Data Preparation](#dataset--data-preparation)
- [Results & Evaluation](#results--evaluation)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Documentation](#documentation)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

This project develops a **coherent, multi-level NLP pipeline** that analyzes podcast episodes by integrating three linguistic dimensions:

1. **Emotion Detection** (Turn-level) — Identify emotional expressions in speaker turns using transformer-based classification
2. **Category Classification** (Episode-level) — Classify podcast episodes into topical categories using aggregated linguistic features
3. **Brand Safety Analysis** (Episode-level) — Detect toxic or unsafe content to support moderation and content filtering

**Core Innovation:** Rather than solving these tasks independently, we develop a **multi-task learning model** with shared representations that learns how emotions, topics, and safety interact across episodes. This demonstrates that different layers of linguistic information are interdependent and can be leveraged to improve overall understanding of podcast content.

**Dataset:** 907 podcast episodes (123,379 speaker turns) from the [SPoRC](https://huggingface.co/datasets/blitt/SPoRC) corpus after filtering for episodes with available speaker turn data.

---

## Team Contributions

| Team Member | Contributions |
|-------------|--------------|
| **Cristina Semikina** | Data preprocessing and dataset exploration, Episode-level category classification, Multi-task model architecture & joint training & evaluation, integration of all three tasks, writing up the report |
| **Gustav Sivel Aakjær Nielsen** | Data preprocessing and dataset exploration, turn-level emotion detection, toxicity analysis, emotional-category-safety correlation analysis including graphs, writing up the report |
| **Oguzhan Sarisakaloglu** | |

**Collaborative Work:**
- Joint design of the multi-task pipeline architecture
- Shared evaluation metrics and performance analysis
- Integrated validation of task interactions

---

## The Multi-Task Pipeline

The pipeline operates by processing each episode through a shared transformer encoder that simultaneously learns three tasks:

- **Turn-level emotion prediction:** Classifies emotional expressions in speaker turns using KL divergence loss against soft labels
- **Episode-level category prediction:** Classifies the primary topic using cross-entropy loss
- **Episode-level toxicity prediction:** Predicts safety score using MSE regression loss

The three losses are combined with learnable weights: $\mathcal{L}_{total} = \lambda_{cat} \mathcal{L}_{cat} + \lambda_{tox} \mathcal{L}_{tox} + \lambda_{emo} \mathcal{L}_{emo}$

### How Task Interaction is Evaluated

**1. Shared Representation Learning**
- The encoder learns a common feature space where turn-level and episode-level patterns co-occur
- Emotions predict both local sentiment and contribute to episode-level safety/category decisions
- Category information constrains emotion predictions (some topics naturally co-occur with specific emotions)

**2. Joint Loss Balancing**
- Weight adjustment reveals which tasks synergize vs. conflict
- Example: If toxicity loss dominates, emotions and toxicity are strongly correlated; if balanced, they provide independent signal

**3. Performance Comparison: Multi-Task vs. Single-Task**
- Single-task models trained separately on each task
- Multi-task model trained jointly with shared encoder
- Metrics compared: Accuracy, F1, MAE, and per-task performance
- Interpretation: If multi-task outperforms, tasks share beneficial representations

**4. Correlation Analysis**
- Turn-level emotion-toxicity correlations (e.g., disgust/anger → higher toxicity)
- Episode-level category-emotion distributions (e.g., News → higher anger, Comedy → higher joy)
- Category-safety relationships (e.g., News/Comedy → higher toxicity than Education/Tech)
- These correlations validate that the model learns meaningful task interactions

### Example Findings

- **Emotion-Toxicity Synergy:** Disgust (0.369 mean toxicity) and Anger (0.359) highly predict unsafe content, while Joy (0.049) predicts safety. Joint learning leverages this predictive power.
- **Category-Emotion Coupling:** News/Comedy episodes skew toward anger/surprise; Educational/Tech toward neutral. The shared encoder captures this coupling.
- **Safety-Category Alignment:** News (13.5% toxicity ratio) and Comedy (12.5%) are higher-risk than Technology (~1%) and Education. Multi-task model balances category classification with safety assessment.

---

## Architecture & Implementation

### Models Used

| Task | Architecture | Input | Output | Loss Function |
|------|--------------|-------|--------|---------------|
| **Emotion** (turn-level) | DistilRoBERTa (pretrained) | Turn text | 7-class probabilities | KL Divergence (soft labels) |
| **Toxicity** (episode-level) | toxic-bert (pretrained) | Turn text (cached) | Aggregated score [0,1] | MSE (regression) |
| **Category** (episode-level) | DistilBERT + linear head | Episode text | 20-class logits | Cross-Entropy |
| **Multi-Task** | Shared DistilBERT + 3 heads | Episode text (turn-windowed) | All three outputs | Weighted sum of above |

### Relevant Details about Implementation 

- **Turn-Level Caching:** Emotion and toxicity computed once per turn using pretrained models, cached to disk to avoid recomputation
- **Episode Aggregation:** Turn-level predictions aggregated using mean pooling (emotions) and max pooling (toxicity)
- **Soft Label Generation:** Emotion soft labels preserve uncertainty; toxicity uses max score as episode-level label
- **Dynamic Windowing:** Long episodes (>32 turns) windowed randomly to balance detail with computational constraints
- **Gradient Accumulation:** 4-step accumulation to achieve effective batch size of 16 on memory-constrained hardware
- **Learnable Task Weights:** Loss weights (λ<sub>cat</sub>, λ<sub>tox</sub>, λ<sub>emo</sub>) are learned during training rather than fixed, allowing the model to discover optimal task importance dynamically


---

## Dataset & Data Preparation

### Data Source
[SPoRC (Speech Podcast Research Corpus)](https://huggingface.co/datasets/blitt/SPoRC) - 1.1M podcast episodes with transcripts, metadata, and speaker information.

### Preprocessing Pipeline
1. **Text Cleaning:** Lowercase, remove music/blank markers, normalize whitespace, remove punctuation
2. **Stopword Removal:** NLTK English stopwords filtered
3. **Turn Merging:** Consecutive speaker turns concatenated for coherent context
4. **Long Turn Splitting:** Turns >500 chars split into 500-char chunks to preserve emotion progression
5. **Final Filtering:** Turns with <2 words removed; episodes with <50 turns excluded

### Final Dataset
- **Episodes:** 907 (after cleanup; episodes with speaker turns in dataset)
- **Turns:** 123,379 turn texts
- **Categories:** 20 unique (from category1); 70 total across 10-level hierarchy
- **Avg Turns/Episode:** 135.5

---

## Results & Evaluation

### Single-Task Baselines
- **Emotion Detection:** 7-class classification on turn level using pretrained DistilRoBERTa
- **Toxicity Detection:** Binary/regression using toxic-bert
- **Category Classification:** 20-class classification on episode level using DistilRoBERTa + linear head


### Multi-Task Performance
Performance metrics and detailed results includes:
- Category classification accuracy and F1-score
- Toxicity prediction MAE (mean absolute error)
- Emotion prediction F1 (macro-averaged)
- Weighted overall score combining all three tasks

---

## Repository Structure

```
project-milestone-p2-group14/
│
├── main.ipynb                          # Main notebook: end-to-end pipeline
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore
│
└── data/                               # Data folder (not included in repo)
    ├── episodeLevelDataSample.jsonl.gz
    ├── speakerTurnDataSample.jsonl.gz
    ├── emotion_cache/                  # Cached emotion predictions
    │   └── emotion_predictions.csv
    └── toxicity_cache/                 # Cached toxicity predictions
        └── toxicity_scores.csv
```

---

## Requirements
- Python 3.10+
- Jupyter Notebook
- Dependencies: See `requirements.txt`

### Installation

Install everything via:

```bash
pip install -r requirements.txt
```
Download data from SPoRC (episodeLevelDataSample.jsonl.gz and speakerTurnDataSample.jsonl.gz): https://huggingface.co/datasets/blitt/SPoRC/tree/main

Or directly from the 'group14' folder in the shared 'NLP 2025 Storage' drive: https://drive.google.com/drive/folders/1V8wOGCScSwad4vQui-ob_hBCX3YrX1pE

Create data/ folder and place episodeLevelDataSample.jsonl.gz and speakerTurnDataSample.jsonl.gz inside it.

The emotion_cache and toxicity_cache folders and underlying files will automatically be created the first time you run the  entire code.

---

## Documentation
- README now clearly describes the entire pipeline and how task interactions are evaluated
- LaTeX report includes detailed methods, results, and analysis sections
- Code comments explain architectural decisions and design patterns

---

## Acknowledgments

This project was developed as part of the Natural Language Processing course at Aarhus University.

We thank:
- The course staff for their guidance and constructive feedback
- The creators of the [SPoRC dataset](https://huggingface.co/datasets/blitt/SPoRC) for providing open-access podcast transcription data
- The Hugging Face community for pretrained models and utilities
