# MNIST-from-Scratch

A minimal, educational implementation of a two-layer neural network for MNIST digit classification—built from pure NumPy without any deep-learning frameworks. Includes data loading from OpenML, modular code organization, and end-to-end training & evaluation.

> **Inspired by:** [Building a Neural Network FROM SCRATCH (no Tensorflow/PyTorch, just numpy & math) by Samson Zhang](https://www.youtube.com/watch?v=w8yWXqWQYmU)

---

## 📂 Repository Structure

```

mnist-from-scratch/
├── model.py             # Network definitions, forward/backward passes
├── train_test.py        # Training loop with epoch-wise logging and prediction helpers & sample visualizations
├── utils.py             # Fetches & splits MNIST from OpenML, One-hot encoding & accuracy computation
├── main.py              # Orchestrator: load → train → evaluate
└── requirements.txt     # Dependencies

````

---

## 🚀 Quick Start

1. **Clone the repo**  
   ```bash
   git clone https://github.com/MyTNinja/mnist-from-scratch.git
   cd mnist-from-scratch

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run training & evaluation**

   ```bash
   python main.py
   ```

You’ll see per-epoch train/dev accuracy logs, a final dev-set accuracy, and a sample digit prediction displayed using Matplotlib.

---

## 🔧 File Overview

* **data\_loader.py**
  Uses `fetch_openml('mnist_784')` to fetch data. Normalizes input and performs train/dev split.

* **model.py**

  * Initializes parameters with He initialization
  * Implements ReLU, softmax, forward/backward propagation
  * Updates weights using SGD

* **train.py**
  Epoch-based training with accuracy logging

* **evaluate.py**
  Predicts labels from trained weights, visualizes example digit and its prediction

* **utils.py**
  Contains helper functions for accuracy computation and one-hot encoding

* **main.py**
  Glue script to load data, train model, display results, and report final accuracy

---

## 🙏 Acknowledgements

* [Samson Zhang's video](https://www.youtube.com/watch?v=w8yWXqWQYmU) — foundational explanation and walkthrough
* [OpenML](https://www.openml.org) for dataset access

---
