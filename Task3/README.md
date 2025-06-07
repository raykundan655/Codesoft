# Iris Flower Classification using Random Forest

This project uses the **Random Forest Classifier** to classify Iris flowers into three species: **Setosa**, **Versicolor**, and **Virginica**, based on their petal and sepal measurements.


## Project Overview

This project focuses on **classifying Iris flowers** into three distinct species — **Setosa**, **Versicolor**, and **Virginica** — using machine learning. The model is trained using the popular **Iris dataset**, which contains measurements of **sepal length**, **sepal width**, **petal length**, and **petal width** for each flower.


## Project Goal

> The goal is to build a machine learning model that can:
- Learn from the sepal and petal measurements.
- Accurately classify each flower into its correct species.
- Evaluate the model's performance using classification metrics.

This dataset is ideal for **introductory classification tasks** and demonstrates the power of supervised learning using **Random Forest Classifier**.


## Dataset

The dataset used is the [Kaggle](https://www.kaggle.com/datasets/arshid/iris-flower-dataset) **Iris dataset**, which contains:
- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`
- `species` (Target)

> Note: The species name is cleaned by removing the "Iris-" prefix.



## Technologies Used

- Python 
- Pandas
- NumPy
- Matplotlib (for future visualization)
- Scikit-learn (ML model and metrics)



##  Model Used

### Random Forest Classifier

- A machine learning algorithm for classification tasks.
- Builds multiple decision trees and combines their outputs using **majority voting**.
- Great for handling complex and non-linear data.


## Evaluation Metrics

- **Accuracy** – Overall correctness of the model.
- **Precision** – Correct positive predictions (minimizing false positives).
- **Recall** – Correctly identified positives (minimizing false negatives).
- **F1 Score** – Harmonic mean of precision and recall.
- **Confusion Matrix** – Visual summary of prediction results.



##  Output

```text
Classification Report:
               presetosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```
 
The model achieved an accuracy of **100%** on the test dataset.
This means the model correctly classified **29 out of 30** Iris flower samples.
