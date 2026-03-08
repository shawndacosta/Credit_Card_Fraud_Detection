# 💳 Credit Card Fraud Detection — Deep Learning Project

# Project Overview

**Credit card fraud detection** is a major challenge for financial institutions, as fraudulent transactions can lead to significant financial losses and security risks for customers.

This project focuses on building a **deep learning neural network** capable of detecting fraudulent credit card transactions using anonymized financial data. The dataset contains transaction features transformed through PCA, along with additional variables such as transaction time and amount.

A key difficulty of this problem lies in the **extreme class imbalance**, as fraudulent transactions represent only a very small fraction of the dataset. 

The objective of this project is to train a **neural network** capable of identifying fraud patterns while effectively handling severe class imbalance.

# 📑 Table of Contents

1. [I. Introduction 🪙](#i-introduction-)
2. [II. Dataset & Exploratory Data Analysis 🔍](#ii-dataset--exploratory-data-analysis-)
3. [III. Data Preprocessing 🛠️](#iii-data-preprocessing-%EF%B8%8F)
4. [IV. Neural Network Architecture 🧠](#iv-neural-network-architecture-)
5. [V. Model Evaluation 📊](#v-model-evaluation-)
6. [VI. Conclusion ✔️](#vi-conclusion-%EF%B8%8F)


# I. Introduction 🪙


**Fraud detection** is a critical application of **artificial intelligence** in the financial sector. With millions of transactions processed daily, manual monitoring is not feasible, making automated detection systems essential.

This project aims to build a deep learning model capable of distinguishing between *legitimate* and *fraudulent* transactions based on historical credit card data.

Because fraudulent transactions represent only a very small portion of the dataset, the modeling process must explicitly account for this imbalance during both training and evaluation.
The project follows a structured **deep learning workflow** including:

- Data exploration and validation
- Data preprocessing
- Neural network design
- Model training
- Model evaluation

# II. Dataset & Exploratory Data Analysis 🔍 

The dataset contains **284,807 credit card transactions** made by European cardholders in **September 2013**, recorded over a two-day period.

## Data Validation

Basic data validation checks were performed to ensure the dataset's consistency.

Using standard inspection techniques, the dataset was verified for **missing values** and **structural issues**. No missing values were detected across any feature column, meaning no imputation step was required.

## Feature Description

The dataset is composed of **31 variables**:

- 28 anonymized features (`V1` to `V28`)
- `Time`
- `Amount`
- `Class` (**target variable**)

The variables `V1` to `V28` result from a **Principal Component Analysis (PCA)** transformation applied to the original transaction features. This transformation anonymizes sensitive financial information while preserving the most informative components of the data.

Two variables remain in their original form:

`Time`

Represents the number of seconds elapsed between each transaction and the first transaction in the dataset.

`Amount` 

Represents the monetary value of the transaction.

## Target Distribution

The distribution of the **target variable** is extremely imbalanced:

- `Normal`: 284,315 
- `Fraud`: 492 

Because of this strong imbalance, ***accuracy*** is not a reliable metric. A model predicting only the majority class could still achieve very high accuracy while failing to detect fraud.

For this reason, the evaluation focuses primarily on ***AUPRC*** (Area Under the Precision-Recall Curve), which is better suited for rare-event detection.

Additional metrics such as ***precision*** and ***recall*** are also considered to better understand the model's behavior.

# III. Data Preprocessing 🛠️

## Reproducibility

To make the experiments more consistent across runs, **random seeds** were fixed for *Python*, *NumPy* and *TensorFlow*.  
This ensures that results remain consistent across multiple runs.

## Feature and Target Separation

The dataset was divided into **training** and **validation** sets using an 80/20 split.

A **stratified** split was used to preserve the original class distribution in both datasets. This ensures that the proportion of fraudulent transactions remains consistent between the training and validation subsets.

## Feature Scaling

Only the `Time` and `Amount` features were standardized with a *StandardScaler*, as the PCA-transformed variables are already normalized.

## Handling Class Imbalance

Because fraudulent transactions are extremely rare, **class weights** were applied during training to give more importance to the minority class.

Combined with the **stratified** split, this helps the model remain sensitive to fraudulent transactions during training.


# IV. Neural Network Architecture 🧠

The predictive model used in this project is a **fully connected neural network** implemented with *TensorFlow* and *Keras*. This type of **architecture** is well suited for tabular data and allows the model to capture nonlinear relationships between transaction features.

The **network** consists of four layers:

- Dense layer (128 neurons)

- Dense layer (64 neurons)

- Dense layer (32 neurons)

- Output layer (1 neuron)

The number of neurons gradually decreases across layers (128 → 64 → 32), allowing the network to progressively learn more compact representations of the data.

The hidden layers use the **ReLU** activation function, which is commonly used in deep learning due to its efficiency and ability to model nonlinear relationships.

The final layer uses a **sigmoid** activation function, which converts the network output into a probability between 0 and 1. This is appropriate for binary classification.

## Regularization Techniques

Two additional components were integrated into the architecture to improve training stability and generalization:

**Batch Normalization**

It stabilizes training by normalizing intermediate activations, which can accelerate convergence.

**Dropout**

It randomly disables a fraction of neurons during training. A dropout rate of 0.2 was used to reduce overfitting while still allowing the network to learn meaningful patterns from the data.

## Training Configuration

The model was trained using the **Adam optimizer** and **binary cross-entropy loss**, which are commonly used for **binary classification** problems.

**Early stopping** was applied to prevent **overfitting** by monitoring validation performance.

A relatively large batch size (512) was used to improve training efficiency on this large dataset.

# V. Model Evaluation 📊

Model performance was evaluated on the **validation dataset** using both **training curves** and classification **metrics**.

## Training Curves

The evolution of **training and validation loss** as well as ***AUPRC*** was monitored during training.

The curves show stable convergence, with validation performance improving before reaching a plateau and no clear signs of overfitting.

*(Training curves can be found in the notebook.)*

## Metrics Performance

| Metric        | Value |
|--------------|----------|
| AUPRC      | ~0.81    |
| Precision | ~0.80    |
| Recall | ~0.83    |
| F1-score | ~0.81    |

These results show that the model is able to detect a large proportion of fraudulent transactions while maintaining a reasonable precision.

In fraud detection systems, this balance is important because missing fraudulent transactions can be costly, while too many false positives may create operational difficulties.

# VI. Conclusion ✔️

This project demonstrates how **deep learning** can be applied to a real-world **fraud detection problem** involving highly imbalanced data.

The final model is able to identify a large proportion of fraudulent transactions while maintaining balanced precision and recall, which is essential for **practical fraud detection systems**.

In practice, fraud detection systems often combine multiple models and continuously retrain them as new transaction data becomes available.

## Possible Improvements

Several directions could further improve the project:

- Threshold optimization for fraud detection
- Testing alternative architectures
- Comparing results with tree-based models such as XGBoost or LightGBM
