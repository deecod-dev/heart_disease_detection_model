# Heart Disease Detection Model

This repository contains a machine learning implementation for heart disease detection using Stochastic Gradient Descent (SGD) and Mini-Batch Gradient Descent optimization techniques. The model utilizes binary classification to predict the presence of heart disease based on input features from a dataset.

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Model Implementation](#model-implementation)
- [Training the Model](#training-the-model)
- [Results and Evaluation](#results-and-evaluation)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Introduction

Heart disease is a significant health issue worldwide. Early detection can help prevent severe health problems. This model aims to accurately predict whether an individual has heart disease using various health metrics.

## Dependencies

This project requires the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn` (for data preprocessing and evaluation)

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib scikit-learn

## Data Preparation

Before training the model, the dataset needs to be prepared:

- **Data Loading**: Load the dataset containing heart disease information.
- **Data Preprocessing**: Normalize the features and split the data into training and validation sets.
- **Feature Selection**: Select relevant features for the model.

## Model Implementation

The model is implemented using a custom neural network structure with the following key functions:

- **initialize_parameters**: Initializes weights and biases.
- **forward_propagation**: Calculates predictions based on input data.
- **backward_propagation**: Computes gradients for weights and biases using backpropagation.
- **update_parameters**: Updates weights and biases based on gradients.
- **binary_cross_entropy**: Computes the loss function for binary classification.

## Optimization Techniques

- **Stochastic Gradient Descent (SGD)**: Updates model parameters using one training example at a time.
- **Mini-Batch Gradient Descent**: Updates parameters using a subset (mini-batch) of training examples.

## Training the Model

The training process involves:

- **Setting Hyperparameters**: Define learning rate, number of epochs, and batch size.
- **Training with SGD and Mini-Batch Gradient Descent**: The model is trained using both techniques, with performance tracked over iterations.
- **Early Stopping and Regularization**: Implemented to avoid overfitting, allowing the model to stop training early based on validation performance.

## Results and Evaluation

The performance of the model is evaluated using metrics such as accuracy, precision, recall, and F1-score. These metrics help assess the effectiveness of the model in predicting heart disease.

## Usage

To use this model:

1. Clone the repository.
2. Prepare your dataset in the specified format.
3. Run the training script to train the model.
4. Evaluate the model using the provided evaluation metrics.

## Conclusion

The heart disease detection model provides a robust approach to identifying individuals at risk of heart disease. The use of SGD and Mini-Batch Gradient Descent optimization techniques enhances the model's performance.

