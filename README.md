# Fashion-MNIST: A Comparative Machine Learning Study

> A hypothesis-driven comparison of a from-scratch Logistic Regression baseline, an RBF-kernel SVM, and a from-scratch Convolutional Neural Network on the Fashion-MNIST image classification task, grounded in exploratory data analysis and validated with 5-fold cross-validation.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243.svg)](https://numpy.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4.2-f89939.svg)](https://scikit-learn.org/)

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Why This Project Is Different](#why-this-project-is-different)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
  - [Phase 0: Exploratory Data Analysis](#phase-0-exploratory-data-analysis)
  - [Phase 1: Hypothesis & Model Selection](#phase-1-hypothesis--model-selection)
  - [Phase 2: Implementation & Evaluation](#phase-2-implementation--evaluation)
  - [Phase 3: Reflection](#phase-3-reflection)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Results in Detail](#results-in-detail)
- [Reproducibility](#reproducibility)
- [Future Work](#future-work)
- [Team & Contributions](#team--contributions)
- [Full Report](#full-report)
- [License](#license)

---

## Overview

This repository holds a full comparative machine learning study on **[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)**, a dataset of 70,000 grayscale 28x28 images spanning 10 clothing categories. Instead of jumping straight into modeling, we followed a data-first, hypothesis-driven approach:

1. We explored the raw data through PCA projections, pixel-intensity distributions, class-wise similarity maps, and intra-class variance grids.
2. We formed a testable hypothesis about which model architecture should win, and why, before training a single model.
3. We implemented three classifiers of increasing complexity, two of them entirely from scratch using only NumPy (no TensorFlow, no PyTorch, no `sklearn` model classes).
4. We tested that hypothesis against hard evidence: accuracy, macro F1, per-class F1, confusion matrices, and 5-fold cross-validation.
5. We reflected on which exploratory signals actually turned out to be predictive, and which ones were misleading in hindsight.

The core question we wanted to answer: does preserving 2D spatial structure actually matter for this dataset, or is a strong non-linear classifier on flattened pixels good enough?

---

## Key Results

| Model | Accuracy | Macro F1 | CV Mean ± Std | Coat F1 | Shirt F1 | Pullover F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression (baseline, from scratch) | 82.08% | 0.8190 | 0.8246 ± 0.0049 | 0.74 | 0.56 | 0.71 |
| SVM, RBF kernel (flexible, scikit-learn) | 90.33% | 0.9027 | 0.9034 ± 0.0035 | 0.85 | 0.73 | 0.84 |
| **CNN (main model, from scratch)** | **92.08%** | **0.9213** | 0.9034 ± 0.0038 | **0.88** | **0.78** | **0.88** |

The headline finding is that a from-scratch NumPy CNN beats a fully-optimized scikit-learn SVM, by 1.75 points in accuracy and 0.0186 in macro F1, mainly because it preserves the 2D spatial layout of the image instead of flattening it into a 784-length vector. The gap is biggest on the exact classes that our Phase 0 EDA flagged as visually ambiguous: Coat, Shirt, and Pullover.

---

## Why This Project Is Different

Most Fashion-MNIST tutorials just open with `model.fit()`. We wanted every modeling decision to be earned from the data instead:

- **EDA comes before modeling.** Ten figures (five illustrative, five deliberately non-illustrative) feed into an evidence table that ties observed data properties directly to specific architecture decisions.
- **From-scratch implementations.** The Logistic Regression and CNN, including convolution, max pooling, dropout, backpropagation, and momentum SGD with learning-rate decay, are all hand-built in NumPy. No autograd, no high-level deep learning framework.
- **Hypotheses are written down before results exist**, then checked explicitly against the final metrics in a dedicated hypothesis-testing section.
- **We report macro F1 and per-class breakdowns**, not just accuracy, since the EDA already showed some classes are inherently harder than others.
- **5-fold cross-validation** on every model, so the results aren't just an artifact of one lucky train/test split.
- **Full transparency on AI usage.** Every instance of AI assistance (debugging, clarification, editing) is logged per phase and per author, with the purpose and impact clearly stated.

---

## Repository Structure

```
fashion-mnist-machine-learning/
│
├── README.md
│
├── models/
│   ├── logistic_regression.py      # Baseline: Logistic Regression from scratch
│   ├── svm.py                      # Flexible: SVM with RBF kernel
│   └── cnn.py                      # Main model: CNN from scratch
│
└── report/
    └── Project_Report.pdf          # Full project report
```

---

## Methodology

### Phase 0: Exploratory Data Analysis

Before choosing any model, we profiled the dataset from a few different angles:

- **PCA projection** (2D) of all 10 classes. This showed that footwear and trousers form tight, well-separated clusters, while upper-body garments (T-shirt, Shirt, Coat, Pullover) overlap heavily.
- **Representative sample grids** (typical, atypical, brightest, darkest per class), which exposed high intra-class variance and lighting sensitivity.
- **Pixel intensity histograms** per class, showing that most images are dominated by background (pixels near zero), and that brightness alone isn't enough to separate classes.
- **Class difference maps**, where a class's mean image is subtracted from the mean of all other classes. These showed that discriminative information is spatially localized (for example, Trouser's two leg columns or Sandal's sole band), not spread evenly across the image.
- **A visual similarity matrix** across all class pairs, which quantitatively identified Coat and Shirt as the hardest pair to separate, and Bag as the easiest class.
- We also included four non-illustrative counter-examples on purpose (sorted pixels, RGB channel analysis on grayscale data, 1D flattened bar charts, brightness vs. index scatter), to show why certain naive analyses fail and what that failure means for model choice.

All of this fed into an evidence table mapping each figure to a concrete modeling implication (for example, "the important pixels are spatially localized" leads directly to "a model must not flatten the image").

### Phase 1: Hypothesis & Model Selection

Based on the EDA, our hypothesis was this: a linear model (Logistic Regression) would underperform on visually overlapping classes because Fashion-MNIST isn't linearly separable in raw pixel space, and a CNN would outperform both the linear model and a flattening-based non-linear model (SVM) because it's the only architecture that preserves 2D spatial structure, which the difference maps showed is where the real discriminative signal lives.

That's what led to the three-model setup:
- **Baseline:** Logistic Regression, to establish the linear-separability floor.
- **Flexible comparison:** SVM (RBF kernel), to test whether a strong non-linear but flattened model could close the gap.
- **Main model:** CNN, to test whether preserving spatial structure is really the deciding factor.

### Phase 2: Implementation & Evaluation

All three models were trained on an 80/20 dev/test split (56,000 and 14,000 images, `random_state=42`, stratified) and evaluated with accuracy, macro F1, per-class precision/recall/F1, and 5-fold cross-validation. See [Models](#models) below for the architecture and hyperparameter details.

### Phase 3: Reflection

The report closes with an honest look back: which EDA signals turned out to be genuinely predictive (the pixel-importance difference maps and the visual similarity matrix both correctly forecast the hardest classes), which ones were a bit misleading (the pixel brightness histograms overstated how much brightness actually mattered, and PCA's 2D view oversimplified true class separability), and how we'd change the pipeline on a second pass (edge and texture-based EDA, a deeper CNN with batch normalization, and data augmentation).

---

## Models

### 1. Baseline: Logistic Regression (from scratch, NumPy only)

- Multinomial softmax regression, weights initialized to zero, a 784 x 10 weight matrix
- Trained with vanilla gradient descent: `lr=0.1`, `epochs=500`
- Manual softmax with max-subtraction for numerical stability
- Manual cross-entropy gradient and accuracy/F1 computation

### 2. Flexible Comparison: SVM (RBF Kernel), via scikit-learn

- `SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr', random_state=42)`
- We chose the RBF kernel specifically to handle the non-linear class overlap the PCA analysis surfaced
- Operates on flattened 784-dimensional pixel vectors, same as Logistic Regression

### 3. Main Model: Convolutional Neural Network (from scratch, NumPy only)

```
Input (N, 28, 28, 1)
  -> Conv1 (16 filters, 3x3, He init) -> ReLU -> MaxPool (2x2)     -> (N, 13, 13, 16)
  -> Conv2 (16 filters, 3x3, He init) -> ReLU                      -> (N, 11, 11, 16)
  -> Flatten                                                       -> (N, 1936)
  -> FC1 (256 units) -> ReLU -> Dropout (p=0.4)
  -> FC2 (10 units) -> Softmax
```

- Every component (convolution forward/backward, max pooling forward/backward, dropout, fully connected layers, backpropagation) is implemented by hand.
- **Optimizer:** Momentum SGD (`momentum=0.9`) with learning-rate decay (`0.95` per epoch, starting at `lr=0.01`)
- **Regularization:** Dropout (`p=0.4`) on the FC1 layer, to reduce overfitting on the visually ambiguous samples we flagged in EDA
- **Training:** 40 epochs, batch size 64
- Loss dropped smoothly from 0.5723 to 0.0616 over the 40 epochs with no instability

---

## Installation

### Prerequisites

| Tool | Version |
|---|---|
| Python | 3.10.x |
| NumPy | 1.26.4 |
| scikit-learn | 1.4.2 |
| matplotlib | 3.8.4 |
| seaborn | 0.13.2 |

### Steps

```bash
git clone https://github.com/<your-username>/fashion-mnist-machine-learning.git
cd fashion-mnist-machine-learning

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install numpy==1.26.4 scikit-learn==1.4.2 matplotlib==3.8.4 seaborn==0.13.2
```

The Fashion-MNIST dataset is fetched automatically through `sklearn.datasets.fetch_openml` the first time you run a script, and cached locally after that. You'll only need an internet connection for that first run.

---

## Usage

Each script is self-contained. It loads the data, trains the model, runs cross-validation, and prints a full classification report.

```bash
# Baseline: Logistic Regression (a few minutes)
python base_model_Logistic_regression.py

# Flexible: SVM with RBF kernel (roughly 10 to 20 minutes)
python flexible_model_svm.py

# Main model: CNN (roughly 30 to 90 minutes on CPU, 8GB+ RAM recommended)
python main_model_cnn.py
```

Each run prints:
- Training progress (per-epoch loss for the CNN, per-fold accuracy for cross-validation)
- Final test-set accuracy and macro F1
- A full per-class precision, recall, F1, and support table
- Cross-validation mean and standard deviation

---

## Results in Detail

Logistic Regression struggled exactly where we expected: Coat (0.74), Shirt (0.56), and Pullover (0.71) were its three weakest classes, which lines up with what the PCA projection flagged as overlapping.

SVM (RBF) closed a lot of that gap by learning curved decision boundaries (Coat 0.85, Shirt 0.73, Pullover 0.84), but it was still capped by working on flattened, spatially blind input.

The CNN beat both models on every single hard class (Coat 0.88, Shirt 0.78, Pullover 0.88), while also matching or beating SVM's cross-validation stability, despite being built entirely without a deep learning framework.

Across all three models, Shirt stayed the hardest class, most often confused with T-shirt. That tracks with the 0.966 visual-similarity score we computed in Phase 0, the highest of any class pair in the dataset.

Easy, structurally distinct classes like Trouser, Sandal, Bag, and Ankle Boot scored around 0.97+ F1 across all three models, matching how well separated their clusters looked in PCA.

---

## Reproducibility

- All train/test splits use `random_state=42` and stratified sampling.
- Cross-validation uses a fixed `k=5` folds, computed the same way across all three scripts.
- Model weight initialization uses fixed seeds where it applies (He initialization for the CNN, zero initialization for Logistic Regression).

---

## Future Work

Based directly on our Phase 3 reflection, here's what we'd want to try next:

- Add a third convolutional layer and batch normalization to the CNN, for more stable, higher-capacity training.
- Apply data augmentation (horizontal flips, brightness jitter) to make the model more robust to the intra-class variation we saw in EDA.
- Swap the brightness-based EDA for edge and texture-based analysis (Sobel, Laplacian) to better explain the confusion between visually similar upper-body classes.
- Benchmark against a framework-based CNN (PyTorch or TensorFlow) to see how much performance is left on the table by keeping everything in NumPy.

---

## Team & Contributions

| Member | Role |
|---|---|
| **Rameen** (2023-EE-3) | Main model (CNN), full from-scratch implementation; EDA figures 1 to 3 and non-illustrative figures 4 to 5; hypothesis write-up; CNN results, hypothesis testing, and reflection sections |
| **Navaal Noshi** (2023-EE-5) | Baseline (Logistic Regression) and flexible (SVM) models, full implementation and evaluation; illustrative EDA figures 4 to 5 and non-illustrative figures 1 to 3; model comparison and reflection sections |

A full per-section contribution breakdown, along with a detailed AI-usage disclosure (tool used, purpose, and impact for every instance of AI assistance across all four project phases), is documented in the [full report](#full-report).

---

## Full Report

The complete write-up, including all EDA figures, the evidence table, hypothesis testing against the final results, confusion matrix analysis, and the full AI usage disclosure, is available in `Project_Report.pdf`.

---



---

*This project was completed as coursework for EE 439: Introduction to Machine Learning, Spring 2026.*
