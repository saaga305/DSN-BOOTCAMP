# Car Price Prediction with CatBoost and Optuna

This repository contains a Kaggle competition notebook for predicting car prices.  
The workflow includes exploratory data analysis (EDA), feature engineering, hyperparameter optimization using Optuna, model training with CatBoost, and generating a final submission file.

---

## Project Overview

Accurately predicting used car prices is important for buyers, sellers, and dealers.  
This project leverages machine learning techniques to build a robust regression model with a focus on:

- Exploratory Data Analysis (EDA)
- Feature engineering for engine and transmission details
- Handling categorical and numerical features
- Hyperparameter optimization with Optuna
- Model training and validation using CatBoost
- Feature importance analysis
- Submission file creation for Kaggle

---

## Workflow

1. **Setup and Imports**  
   Install and import required Python libraries.

2. **Load Data**  
   Load the `train.csv`, `test.csv`, and `sample_submission.csv` files.

3. **Exploratory Data Analysis (EDA)**  
   - View dataset shape, data types, and missing values  
   - Visualize price distribution  
   - Generate correlation heatmap  
   - Explore unique counts and categorical variables

4. **Preprocessing and Feature Engineering**  
   - Handle missing values for categorical and numerical features  
   - Frequency encoding for brand  
   - Extract engine and transmission-related features  
   - Create new features such as age, mileage per year, log mileage, and interaction terms

5. **Model Training with Optuna**  
   - Define an objective function for hyperparameter optimization  
   - Run Optuna to minimize validation RMSE  
   - Use CatBoostRegressor for training

6. **Validation**  
   - Split data into train/validation sets  
   - Evaluate the model on validation data  
   - Compute and print RMSE score

7. **Feature Importance**  
   - Visualize top 20 most important features from CatBoost

8. **Final Model and Submission**  
   - Retrain on full training data with best parameters  
   - Predict prices on test data  
   - Save results in `submission_enhanced_plus.csv`

---

## Requirements

The following Python libraries are required:

```bash
pip install catboost optuna seaborn matplotlib
