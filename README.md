# Neural Network from Scratch — 2D Nonlinear Classification

This mini-project is a full implementation of a feedforward neural network (multilayer perceptron) in Python — entirely **from scratch** using only built-in libraries and `numpy`. It solves a two-dimensional **nonlinear binary classification problem** through supervised learning.

The mini-project was created as part of the COEN 6331 Neural Networks course at Concordia University.

## 📌 Features

- Pure Python (no TensorFlow, no PyTorch)
- Modular neural network:
  - Variable number of hidden layers
  - Custom neurons per layer
  - Choice of activation functions (ReLU, Sigmoid, Leaky ReLU)
- From-scratch implementations of:
  - Forward and backward propagation
  - Gradient descent and mini-batch stochastic gradient descent
  - Adam optimizer
  - Binary cross-entropy loss
- Data generation, normalization, and labeling
- Visualization of:
  - Learning curves (accuracy, loss)
  - Decision boundaries
  - Confusion matrix

## Problem Description

The task is to classify 2D points into two nonlinear regions, **C₁** and **C₂**, based on their `(x₁, x₂)` coordinates. The points are uniformly generated in polar coordinates and labeled according to the underlying region they fall into.

This is a **non-linearly separable problem**, hence simple linear models like single-layer perceptrons are insufficient. The project demonstrates how increasing complexity through deeper networks and tuning hyperparameters improves classification accuracy.

While this project demonstrates a fully custom neural network implementation, it's worth noting that simpler and more efficient methods (e.g., logistic regression, support vector machines, or even k-nearest neighbors) can also solve this classification problem effectively, especially given the clean and synthetic nature of the data.

However, the purpose of this project is not to choose the most optimal algorithm, but to build neural networks from first principles and explore their behavior in a controlled scenario. It serves as a foundational exercise in understanding backpropagation, gradient descent, and model architecture.

## Simulation Overview

The notebook explores several configurations:

- **Single-layer perceptron**: poor performance (≈75%)
- **Single hidden layer MLP**: better accuracy with more neurons
- **Deep MLP ([2, 50, 25, 1])**: excellent performance (≈99.5%)
- **Optimizers**:
  - Vanilla SGD
  - Mini-batch SGD
  - Adam (best convergence)

### Best Configuration
```
| Parameter           | Value                |
|---------------------|----------------------|
| Architecture        | [2, 50, 25, 1]       |
| Optimizer           | Adam                 |
| Learning rate       | 0.005                |
| Batch size          | 500                  |
| Accuracy (Test set) | ~99.5%               |
```

## Project Structure

mlp-2d-classifier/
├── main.ipynb # Main Jupyter notebook with full implementation and results
├── Report (academic).pdf # Assignment report
└── README.md

## Requirements

- Python 3.x
- NumPy
- Matplotlib

Install with:

```bash
pip install numpy matplotlib
```

## Learning Outcomes
- Understanding the role of depth, width, and activation in neural nets
- Importance of data normalization and initialization
- Effectiveness of various optimizers and batch sizes
- Mastery of the backpropagation algorithm

## Visual Outputs
The notebook includes plots of:
- Training and validation curves
- Test set decision boundaries
- Accuracy vs. neurons/layers/learning rates
- Confusion matrix and probability maps

## License
This project is provided for academic purposes. Feel free to explore, modify, or build upon it.
