<div align="center">

# 🔢 Handwritten Text Recognition (OCR)

**A deep-learning pipeline for recognising handwritten digits and letters — powered by CNN models trained on MNIST & EMNIST, with an interactive Streamlit demo.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training](#1-training)
  - [Evaluation](#2-evaluation)
  - [Streamlit Demo](#3-streamlit-demo-app)
- [Preprocessing & Augmentation](#-preprocessing--augmentation)
- [Results](#-results)
- [Authors](#-authors)
- [License](#-license)

---

## 🧠 Overview

This repository contains a complete **Optical Character Recognition (OCR)** system that recognises handwritten digits (0–9) and letters (A–Z) from images. The system is built around Convolutional Neural Network (CNN) models trained on the **MNIST** (digits) and **EMNIST Letters** (letters) datasets, and exposes predictions through an interactive **Streamlit** web application.

Three recognition modes are supported:

| Mode | Description |
|------|-------------|
| **Single digit** | Classifies a single handwritten digit with top-3 confidence scores |
| **Single letter** | Classifies a single handwritten letter (A–Z) |
| **Multi-symbol / Mixed** | Segments a row of handwritten characters and returns the full recognised string |

---

## ✨ Features

- 🖼️ **Image preprocessing** — normalisation, binarisation (Otsu thresholding), and augmentation
- 🔢 **Digit recognition** — CNN trained on MNIST achieving **>99 % accuracy**
- 🔡 **Letter recognition** — CNN trained on EMNIST Letters (A–Z)
- 🔀 **Mixed-mode pipeline** — joint digit + letter segmentation and recognition
- 📊 **Training analytics** — loss/accuracy curves, confusion matrix, misclassified examples
- 🌐 **Interactive Streamlit app** — upload any image and get instant predictions

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | ![Python](https://img.shields.io/badge/-Python_3.8+-3776AB?logo=python&logoColor=white) |
| Deep Learning | ![TensorFlow](https://img.shields.io/badge/-TensorFlow_2.x-FF6F00?logo=tensorflow&logoColor=white) · ![Keras](https://img.shields.io/badge/-Keras-D00000?logo=keras&logoColor=white) |
| Computer Vision | ![OpenCV](https://img.shields.io/badge/-OpenCV_4.9-5C3EE8?logo=opencv&logoColor=white) |
| Data & ML | ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) · ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) · ![scikit--learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white) |
| Visualisation | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c) · ![Seaborn](https://img.shields.io/badge/-Seaborn-4C72B0) |
| Demo App | ![Streamlit](https://img.shields.io/badge/-Streamlit_1.39-FF4B4B?logo=streamlit&logoColor=white) |

---

## 📁 Project Structure

```
OCR/
├── src/
│   └── ocr/
│       ├── models/
│       │   └── cnn.py              # CNN architecture definition
│       ├── train/
│       │   ├── train_mnist.py      # Digit model training script
│       │   └── train_emnist_letters.py  # Letter model training script
│       ├── eval/
│       │   ├── evaluate.py         # Digit model evaluation
│       │   ├── evaluate_letters.py # Letter model evaluation
│       │   ├── plot_training.py    # Loss/accuracy curve plots
│       │   └── plot_samples.py     # Sample & misclassified image plots
│       ├── data/                   # Data loading utilities
│       ├── baselines/              # Baseline model experiments
│       ├── pipeline.py             # Multi-symbol segmentation & prediction
│       └── preprocess.py           # Image preprocessing utilities
├── artifacts/                      # Saved models, logs, plots
├── notebookForOcr/
│   └── notebookOCR.ipynb           # Exploratory Jupyter notebook
├── samples/                        # Sample input images
├── streamlit_app.py                # Streamlit web application
├── requirements.txt                # Pinned dependencies
├── requirements.in                 # Top-level dependencies
└── TASKS.md                        # Sprint task board
```

---

## ✅ Prerequisites

- **Python 3.8+**
- *(Optional)* NVIDIA GPU with CUDA drivers for accelerated training
- A virtual environment manager (`venv` or `conda`)

---

## 🚀 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/teodora525/OCR.git
   cd OCR
   ```

2. **Create and activate a virtual environment**

   ```bash
   # Linux / macOS
   python -m venv .venv
   source .venv/bin/activate

   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 💡 Usage

### 1. Training

Train the **digit** model (MNIST):

```bash
python src/ocr/train/train_mnist.py --output-dir artifacts/
```

Train the **letter** model (EMNIST Letters):

```bash
python src/ocr/train/train_emnist_letters.py --output-dir artifacts/
```

<details>
<summary>Common training arguments</summary>

| Argument | Description |
|----------|-------------|
| `--output-dir` | Directory where the model and logs are saved |
| `--config` | Path to a YAML/JSON config file (model, optimiser, lr, batch size, augmentations) |
| `--resume` | Path to a checkpoint to resume training from |
| `--gpu` | GPU ID(s) to use |

</details>

### 2. Evaluation

Evaluate the digit model and generate a confusion matrix:

```bash
python src/ocr/eval/evaluate.py
```

Evaluate the letter model:

```bash
python src/ocr/eval/evaluate_letters.py
```

Plot training loss/accuracy curves from the saved CSV log:

```bash
python src/ocr/eval/plot_training.py
```

### 3. Streamlit Demo App

```bash
streamlit run streamlit_app.py
```

Open the URL printed in the terminal (default: `http://localhost:8501`) and:

1. *(Optional)* Check **Letters mode** to switch to A–Z recognition.
2. *(Optional)* Check **Multi-symbol mode** for segmentation of a sequence of characters.
3. *(Optional)* Check **Mixed mode** for combined digit + letter recognition (enabled by default).
4. Upload a PNG/JPG image and view the prediction with confidence scores.

---

## 🔬 Preprocessing & Augmentation

The pipeline applies the following techniques before inference and during training:

- **Resize / height normalisation** — preserves aspect ratio to 28 × 28 input
- **Binarisation** — Otsu's adaptive thresholding for clean foreground extraction
- **Normalisation** — pixel values scaled to [0, 1]
- **Augmentation (training only)**
  - Small random rotations
  - Random brightness & contrast jitter
  - Elastic distortions
  - Random cropping & noise

---

## 📈 Results

| Model | Dataset | Test Accuracy |
|-------|---------|--------------|
| CNN (digits) | MNIST | **> 99 %** |
| CNN (letters) | EMNIST Letters | see `artifacts/` |

Evaluation artefacts (confusion matrix, misclassified samples, training curves) are saved under `artifacts/` after running the evaluation scripts.

---

## 👥 Authors

| GitHub | Role |
|--------|------|
| [@paniicj0](https://github.com/paniicj0) | Model architecture, training pipeline, backend |
| [@teodora525](https://github.com/teodora525) | Evaluation, visualisation, Streamlit frontend |

For questions or bug reports please [open an issue](https://github.com/teodora525/OCR/issues) on GitHub.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ by <a href="https://github.com/paniicj0">paniicj0</a> &amp; <a href="https://github.com/teodora525">teodora525</a>
</div>

