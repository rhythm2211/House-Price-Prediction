# House-Price-Prediction

Project by: Rhythm Suthar

# House Price Prediction Project

## Introduction

Welcome to the House Price Prediction Project! This repository contains code and documentation for predicting house prices using advanced machine learning techniques. By leveraging Support Vector Regressor (SVR), Random Forest Regressor, and Gradient Boosting Regressor (GBR) models, this project aims to develop an accurate predictive model for house prices based on various property features.

## Objective

The primary objective of this project is to build and evaluate different regression models to predict house prices accurately. The models are tested on unseen data to identify the best-performing model for reliable house price predictions.

## Methodology

The project follows these steps:

1. **Data Preprocessing**:
   - Handle missing values using imputation techniques.
   - Encode categorical variables using OneHotEncoder.
   - Apply polynomial features transformation to capture non-linear relationships.
   - Scale the features to standardize the data.

2. **Model Training**:
   - Train three different regression models (SVR, Random Forest, and GBR) on the preprocessed training data.
   - Perform hyperparameter tuning using GridSearchCV to find the optimal parameters for each model.

3. **Model Evaluation**:
   - Evaluate the performance of each model on unseen test data.
   - Use metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) to compare the models' performance.

4. **Prediction and Submission**:
   - Make predictions using the best-performing model.
   - Format and save the results as a CSV file for submission.

## Tools and Libraries

The following tools and libraries are used in this project:

- **Python**: The core programming language used for the implementation.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning model implementation and evaluation.
- **Matplotlib & Seaborn**: For data visualization.

## Usage

1. Open the Jupyter notebook:
    ```bash
    jupyter notebook
    ```
2. Navigate to the notebook file and open it.

3. Follow the steps in the notebook to preprocess the data, train the models, evaluate their performance, and make predictions.

## Results

After thorough evaluation, the best model among SVR, Random Forest, and GBR for house price prediction is identified. The results demonstrate the effectiveness of machine learning models in predicting complex real-world phenomena such as house prices.

## Acknowledgments

- This project was created as part of the Celebal Summer Internship.
- This project uses data from : https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

