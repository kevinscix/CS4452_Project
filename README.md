# Multimodal Classification of Harmful Memes via Image-Text Fusion

**Group Members:**
- Kevin Wu
- Dipraman Ghosh
- Raghav Gulati
- Aryan Malhotra

## Project Overview
Harmful meme detection requires jointly understanding an image and any embedded or implied text, making it a challenging multimodal deep learning problem. In this project, we build a classifier that predicts whether a meme is harmful by fusing features from an image encoder (e.g., ResNet or ViT) and a text encoder (e.g., DistilBERT).

We evaluate three model variants:
1. **Image-only**
2. **Text-only** 
3. **Multimodal fusion**

Performance is compared using Accuracy, F1-score, and ROC-AUC metrics. The project also includes qualitative error analysis to identify failure modes such as sarcasm and context dependence.

---

## Dataset Structure (Consolidated)

The dataset has been consolidated and cleaned for this project. The original split files have been moved to `old data/` to keep the workspace organized.

- **`img/`**: Contains the complete set of 11,707 PNG images used in this project.
- **`master_list.jsonl`**: The master metadata file containing 11,707 unique records. Every record has been verified to have an existing image and a ground-truth label.
- **`old data/`**: Contains the original competition split files (`train.jsonl`, `dev_seen.jsonl`, etc.) and legacy raw data for reference.

### Metadata Format (.jsonl)
Each line is a JSON object with the following fields:
- `id`: Unique 5-digit identifier (e.g., `"01256"`)
- `img`: Path to the image (e.g., `"img/01256.png"`)
- `label`: Binary label (0 = not-hateful, 1 = hateful)
- `text`: The text occurring in the meme

---

## Getting Started

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Data
The `master_list.jsonl` needs to be split into train, validation, and test sets before training:
```bash
python split_data.py
```
This will create a `data/` directory with the split files.

### 3. Training the Model
To avoid module import issues, always run the scripts as modules using the `-m` flag from the project root:

**Multimodal (Default):**
```bash
python -m src.train --data_dir data --epochs 5 --batch_size 16
```

**Text-only:**
```bash
python -m src.train --model_type text --data_dir data --epochs 5 --batch_size 16
```

**Image-only:**
```bash
python -m src.train --model_type image --data_dir data --epochs 5 --batch_size 16
```

### 4. Evaluation
Evaluate a trained model checkpoint:
```bash
python -m src.eval --model_type multimodal --model_path model_multimodal_best.pt --data_dir data
```

---

## Original Dataset Reference
This project uses the **Hateful Memes Challenge** dataset created by Facebook AI.

- **Paper:** [The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes](https://arxiv.org/abs/2005.04790)
- **Website:** [hatefulmemeschallenge.com](https://hatefulmemeschallenge.com)

### License
The dataset is licensed under the terms in the `LICENSE.txt` file.

### Citations
```bibtex
@inproceedings{Kiela:2020hatefulmemes,
 author = {Kiela, Douwe and Firooz, Hamed and Mohan, Aravind and Goswami, Vedanuj and Singh, Amanpreet and Ringshia, Pratik and Testuggine, Davide},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {2611--2624},
 publisher = {Curran Associates, Inc.},
 title = {The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes},
 year = {2020}
}
```
