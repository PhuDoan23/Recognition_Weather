# A2WNet: Adaptive Attention Weather Network

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A2WNet** is a hybrid CNN–Transformer architecture for weather image recognition. It fuses local spatial features from VGG16 with global context from ViT-B/16 through a **dynamic gating mechanism** and trains with **supervised contrastive loss** to produce discriminative weather embeddings.
>
> 📄 *Accepted at [ICDAR 2026]*


---

## Key Results

All models trained with **5 random seeds** on the [Weather Phenomenon Database (WEAPD)](https://doi.org/10.7910/DVN/M8JQCR) (6,863 images, 11 classes). Results show **mean ± std** test accuracy:

| Model | Test Accuracy | Std | Parameters | Key Strength |
|---|---|---|---|---|
| **A2WNet (Ours)** | **92.34%** | ±0.40% | ~90M | Highest consistency, contrastive separation |
| ViT-B/16 | 92.29% | ±0.50% | 86M | Strong global features |
| HybridGated | 92.00% | ±0.72% | ~90M | Dynamic feature fusion |
| VGG16 | 90.40% | ±0.51% | ~15M | Robust local features |
| ResNet50 | 90.68% | — | 25M | Best accuracy-to-parameter ratio |
| MobileNetV2 | 85.36% | — | 3.4M | Lightest for edge deployment |
| AlexNet | 75.75% | — | 60M | Baseline (trained from scratch) |

### ImageNet Ablation Study

| Comparison | Δ Mean Accuracy | p-value |
|---|---|---|
| VGG16 (ImageNet) vs VGG16 (scratch) | +23.03% | < 0.001 |
| A2WNet (ImageNet) vs A2WNet (scratch) | +27.12% | < 0.001 |

*Transfer learning provides a statistically significant boost (paired t-test, n=5 seeds).*

---

## Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with ≥ 12 GB VRAM (recommended for hybrid models)
- CUDA 12.x + cuDNN

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/PhuDoan23/Recognition_Weather.git
cd Recognition_Weather

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Download the [Weather Phenomenon Database (WEAPD)](https://doi.org/10.7910/DVN/M8JQCR) and organize it as:

```
dataset/
├── dew/
├── fogsmog/
├── frost/
├── glaze/
├── hail/
├── lightning/
├── rain/
├── rainbow/
├── rime/
├── sandstorm/
└── snow/
```

> **Note:** The `dataset/` directory is excluded from version control via `.gitignore`.

---

## Reproducing Results

### Train All Baseline Models (Sequential)

```bash
python scripts/train_all.py
```

This trains VGG16 → AlexNet → ResNet50 → MobileNetV2 → ViT one after another, logging results to `logs/`.

### Multi-Seed Experiment (4 models × 5 seeds)

```bash
# Run all 20 training runs
python scripts/train_seeds.py

# Run for a specific model only
python scripts/train_seeds.py --model a2wnet
```

Results are saved to `results/runs/{model}_seed{seed}.json`.

### Ablation Study (No ImageNet Weights)

```bash
# Run all 10 ablation runs
python scripts/train_ablation.py
```

### Statistical Analysis

```bash
# Pretrained models only
python scripts/analyze_seeds.py

# Pretrained + ablation comparison
python scripts/analyze_seeds.py --all
```

### Visualization

```bash
# t-SNE comparison of latent spaces
python src/plot_tsne.py

# Grad-CAM heatmaps (VGG16 vs A2WNet)
python src/test_gradcam.py --image <path_to_image> --output figures/gradcam_output.png
```

---

## Project Structure

```
Recognition_Weather/
├── src/                          # Model architectures
│   ├── __init__.py               # Package exports
│   ├── baselines/                # Standard architectures
│   │   ├── vgg16.py
│   │   ├── alexnet.py
│   │   ├── resnet.py
│   │   ├── mobilenet.py
│   │   └── vit.py
│   ├── contributions/            # Hybrid proposed models
│   │   ├── hybrid_vgg_vit.py     # Dual-branch hard concat fusion
│   │   ├── hybrid_gated.py       # Dynamic gating fusion
│   │   └── hybrid_contrastive.py # A2WNet: gating + contrastive loss
│   ├── plot_tsne.py              # t-SNE visualization script
│   └── test_gradcam.py           # Grad-CAM comparison script
├── scripts/                      # Training & analysis scripts
│   ├── train_all.py              # Sequential baseline training
│   ├── train_one.py              # Single model + seed training
│   ├── train_seeds.py            # Multi-seed orchestrator
│   ├── train_ablation.py         # Ablation orchestrator
│   ├── train_ablation_one.py     # Single ablation run
│   └── analyze_seeds.py          # Statistical analysis
├── notebooks/                    # Jupyter notebooks
│   ├── hybrid_model_evaluation.ipynb
│   └── loading_analysis_model.ipynb
├── models/                       # Saved model weights (.h5)
├── utils.py                      # Shared data loading & plotting
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md
```

---

## Infrastructure

| Component | Detail |
|---|---|
| **GPU** | NVIDIA RTX 5070 Ti (16 GB VRAM) |
| **Framework** | TensorFlow 2.21, Keras 3.13, Keras-Hub 0.26 |
| **Optimizer** | AdamW (weight decay = 1e-4) |
| **Training** | Two-phase: frozen backbone → fine-tune top blocks |
| **Callbacks** | EarlyStopping + ModelCheckpoint (save_weights_only) |

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{doan2026a2wnet,
  title     = {A2WNet: Adaptive Attention Weather Network with Supervised Contrastive Learning},
  author    = {Nhat-Tung Le, Doan Phu, Phu Le, and Thai-Thinh Dang},
  booktitle = {Proceedings of [ICDAR 2026]},
  year      = {2026}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [Weather Phenomenon Database (WEAPD)](https://doi.org/10.7910/DVN/M8JQCR) by Xiao, H. (2021), Harvard Dataverse, V1
- [ViT-B/16 ImageNet-21k](https://keras.io/api/keras_hub/) pretrained weights via Keras Hub
- [VGG16](https://arxiv.org/abs/1409.1556) pretrained on ImageNet via Keras Applications
