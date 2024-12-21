# DeepLearningProject

## Overview
This repository contains two scripts implementing binary classification models using two distinct loss functions: Binary Cross Entropy (BCE) and Cross Entropy (CE). Both models are built and trained using TensorFlow/Keras and aim to showcase the differences and practical applications of these loss functions in classification tasks.
The repository's scripts were implemented as part of a project aimed at detecting AI-generated images using fine-tuned Convolutional Neural Networks (CNNs). This project emphasizes model interpretability using GradCAM and uses the CIFAKE dataset to benchmark performance.
### Files Included
- **Binary\_cross\_entropy\_model.ipynb**: A notebook implementing a binary classification model optimized with Binary Cross Entropy loss.
- **Cross\_entropy\_model.ipynb**: A notebook implementing a classification model optimized with Cross Entropy loss.
## Prerequisites
To run these notebooks, ensure you have the following installed:
- Python 3.8 or later
- Jupyter Notebook or Jupyter Lab
- TensorFlow 2.6 or later
- NumPy
- Matplotlib
Install the required libraries using:
```bash
pip install tensorflow numpy matplotlib
```
## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/frmarker/DeepLearningProject.git
   cd DeepLearningProject
   ```
2. Open the notebooks using Jupyter:
   ```bash
   jupyter notebook
   ```
3. Run each cell sequentially to train and evaluate the models.
## Model Details
### Binary Cross Entropy Model
**File**: `Binary_cross_entropy_model.ipynb`
This notebook:
- Uses the Binary Cross Entropy loss function, suitable for binary classification tasks.
- Implements a neural network with a single output neuron and sigmoid activation.
- Demonstrates training, validation, and performance evaluation.
#### Key Components:
- Data preprocessing to prepare binary labels.
- Model architecture:
  - Input layer
  - Hidden layers with ReLU activation
  - Output layer with Sigmoid activation
- Metrics: Accuracy and loss are tracked during training.
### Cross Entropy Model
**File**: `Cross_entropy_model.ipynb`
This notebook:
- Uses the Cross Entropy loss function, suitable for multi-class classification.
- Implements a neural network with multiple output neurons and softmax activation.
- Demonstrates training, validation, and performance evaluation.
#### Key Components:
- Data preprocessing for multi-class labels.
- Model architecture:
  - Input layer
  - Hidden layers with ReLU activation
  - Output layer with Softmax activation
- Metrics: Accuracy and loss are tracked during training.
## Results
Both models include visualizations of:
- Training and validation accuracy over epochs.
- Training and validation loss over epochs.
The project achieved a test accuracy of 95.2% with Cross Entropy and 94.7% with Binary Cross Entropy. GradCAM visualizations were used to interpret the model decisions, focusing on regions in the image that influenced the predictions.
---
