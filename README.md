# Weather Image Recognition

A comprehensive deep learning project for classifying weather conditions from images using state-of-the-art Convolutional Neural Networks (CNNs) and Vision Transformers (ViT). This repository implements, compares, and fuses various architectures to achieve high-accuracy weather recognition.

## 🌦️ Project Overview

This project aims to automate the identification of 11 different weather categories. It explores the transition from classical CNN architectures to modern Transformer-based approaches and culminates in **Hybrid Fusion Models** that leverage both local spatial features (CNNs) and global context (Transformers).

### Dataset Summary
- **Total Images:** 6,863
- **Classes (11):** `dew`, `fogsmog`, `frost`, `glaze`, `hail`, `lightning`, `rain`, `rainbow`, `rime`, `sandstorm`, `snow`.
- **Split:** 64% Train / 16% Validation / 20% Test (Stratified).
- **Augmentation:** Rotation, zoom, shifts, shear, and horizontal flips.

---

## 🏗️ Implemented Architectures

### 1. Classical CNNs
- **AlexNet:** Built from scratch with Batch Normalization and Dropout.
- **VGG16:** Pretrained on ImageNet with a custom dense head.
- **ResNet50:** Residual learning framework for better gradient flow.
- **MobileNetV2:** Lightweight model optimized for mobile/edge deployment.

### 2. Vision Transformer (ViT)
- **ViT-B/16:** Pretrained on ImageNet-21k. Processes images as a sequence of 196 patches using self-attention.

### 3. Hybrid Models (Advanced)
- **Hybrid VGG-ViT:** A dual-branch architecture that concatenates features from VGG16 and ViT-B/16 before classification.
- **Hybrid Gated:** Implements a gating mechanism to dynamically weigh features from different branches.
- **Contrastive Hybrid:** Uses contrastive loss to learn more discriminative weather features.

---

## 📈 Performance Comparison

The models were trained on an NVIDIA RTX 5070 Ti (16GB VRAM).

| Model | Test Accuracy | Test Loss | Params | Key Strength |
|---|---|---|---|---|
| **ViT-B/16** | **93.37%** | **0.266** | 86M | Highest accuracy via ImageNet-21k pretraining |
| **VGG16** | 91.77% | 0.306 | 138M | Robust feature extraction |
| **ResNet50** | 90.68% | 0.409 | 25M | Best accuracy-to-parameter ratio |
| **MobileNetV2** | 85.36% | 0.438 | 3.4M | Extremely lightweight and fast |
| **AlexNet** | 75.75% | 0.706 | 60M | Baseline for training from scratch |

*Note: Hybrid models typically match or exceed ViT-B/16 performance by fusing complementary features.*

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA GPU (Recommended for training)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Recognition_Weather.git
   cd Recognition_Weather
   ```
2. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```

### Usage

#### Training
- **Train All Models (Sequential):**
  ```bash
  python train_all.py
  ```
- **Train a Specific Model (e.g., Hybrid VGG-ViT):**
  ```bash
  python src/Hybrid_VGG_ViT.py
  ```

#### Evaluation
Explore the results and confusion matrices in the `notebook/` directory:
- `hybrid_model_evaluation.ipynb`: Detailed metrics for the fusion models.
- `loading_analysis_model.ipynb`: General model analysis and visualization.

---

## 📂 Project Structure

```text
.
├── dataset/                # Raw weather images organized by class
├── src/                    # Model definitions and specialized training scripts
│   ├── AlexNet.py
│   ├── ViT.py
│   ├── Hybrid_VGG_ViT.py   # Dual-branch fusion model
│   └── ...
├── models/                 # Saved .h5 weights for best performing models
├── logs/                   # Training logs for each model
├── Img/                    # Visualization results (Confusion Matrices, Grad-CAM)
├── plots/                  # Training history plots
├── train_all.py            # Master script for sequential training
└── utils.py                # Data generators and shared helper functions
```

---

## 🛠️ Infrastructure & Frameworks
- **Hardware:** NVIDIA RTX 5070 Ti (Blackwell Architecture)
- **Frameworks:** TensorFlow 2.21, Keras 3.13, Keras-Hub
- **Optimization:** AdamW with Weight Decay, Early Stopping, and Learning Rate Scheduling.

---
