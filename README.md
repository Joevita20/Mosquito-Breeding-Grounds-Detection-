# Mosquito Breeding Grounds Detection

> **Research Paper:** *Mosquito Breeding Grounds Detection Using Deep Learning Techniques*  
> **Authors:** Varalakshmi Perumal, R.Sasana, Rakshitha P, **F. Joevita Faustina Doss**  
> Anna University, MIT Campus

---

## 🎯 Project Goal

Detect mosquito breeding sites from UAV (drone) imagery using multiple deep learning architectures. The 6 target classes are:

| Class | Description |
|---|---|
| `tires` | Old tires that collect stagnant water |
| `water_tanks` | Exposed water storage tanks |
| `bottles` | Discarded plastic/glass bottles |
| `buckets` | Buckets holding water |
| `pools` | Natural/artificial pools |
| `water_tubes` | Water pipes and containers |

---

## 📊 Paper Results (Table I)

| Model | Train Time | Precision | Recall | F1 | mAP |
|---|---|---|---|---|---|
| **YOLOv8** | 110 min | **0.94** | 0.86 | 0.90 | **0.92** |
| Detectron2 | 60 min | 0.85 | 0.80 | 0.82 | 0.85 |
| YOLOv5 | 91 min | 0.88 | 0.74 | 0.80 | 0.82 |
| DETR | 140 min | 0.80 | 0.75 | 0.77 | 0.82 |
| Faster-RCNN | — | 0.67 | 0.90 | 0.77 | 0.71 |

> **Best model: YOLOv8 with SGD optimizer, achieving mAP = 0.92**

---

## 📁 Project Structure

```
Mosquito-Breeding-Grounds-Detection/
├── data/
│   ├── raw/               # Original MBG videos & annotations
│   ├── frames/            # Extracted video frames
│   ├── augmented/         # Augmented images (640×640)
│   ├── gan_generated/     # GAN-synthesized pool images
│   ├── yolo_format/       # YOLO-ready dataset (train/val/test)
│   ├── detectron2_format/ # COCO format for Detectron2
│   └── data.yaml          # YOLO dataset config
├── docs/
│   └── MBG_ML_Updated version_21_5_23.pdf
├── models/
│   ├── yolov8/    train_yolov8.py
│   ├── yolov5/    train_yolov5.py
│   ├── detectron2/ train_detectron2.py
│   └── detr/      train_detr.py
├── notebooks/
│   └── MBG_Training_Comparison.ipynb  ← Main Colab Notebook
├── src/
│   ├── preprocessing/
│   │   ├── extract_frames.py       # Video → images
│   │   ├── augment_data.py         # Paper augmentations
│   │   ├── convert_to_yolo_format.py # COCO → YOLO
│   │   └── gan_gen.py              # DCGAN for minority classes
│   └── evaluation/
│       ├── metrics.py              # mAP, Precision, Recall, F1
│       └── plot_results.py         # Confusion matrix, charts
├── tests/
│   └── test_preprocessing.py
└── requirements.txt
```

---

## 🚀 Quick Start (Google Colab)

1. **Open the main notebook:**  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joevite20/Mosquito-Breeding-Grounds-Detection-/blob/main/notebooks/MBG_Training_Comparison.ipynb)

2. **Follow the notebook steps:**
   - Step 0: Mount Drive & clone repo
   - Step 1: Install dependencies
   - Step 2: Download MBG dataset
   - Step 3: Augment data
   - Step 4: Train GAN (for pool class balance)
   - Steps 6a–6d: Train all four models
   - Step 7: Compare results vs. paper

3. **Get the MBG Dataset:**  
   Download from: https://www02.smt.ufrj.br/~tvdigital/database/mosquito/page_01.html

---

## ⚙️ Local Installation

```bash
git clone https://github.com/Joevita20/Mosquito-Breeding-Grounds-Detection-.git
cd Mosquito-Breeding-Grounds-Detection
pip install -r requirements.txt
```

### Run Tests
```bash
python -m pytest tests/ -v
```

---

## 🧠 Methodology

1. **Dataset:** MBG database (Aedes aegypti UAV videos) split into frames and annotated across 6 classes.
2. **Preprocessing:** 640×640 resize, horizontal/vertical flips, grayscale (4%), saturation (±25%).
3. **Class Imbalance:** GAN used to generate synthetic images for the under-represented `pools` class.
4. **Models:** YOLOv8 (SGD), YOLOv5 (SGD), Detectron2 (Adam/Adagrad), DETR (AdamW).
5. **Evaluation:** mAP@50, Precision, Recall, F1-score.

---

## 📄 License

MIT License — See [LICENSE](LICENSE)
