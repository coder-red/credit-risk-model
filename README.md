<p align="center">
  <img src="assets\Credit_img.png" alt="Project Banner" width="100%">
</p>


![Python version](https://img.shields.io/badge/Python%20version-3.10%2B-lightgrey)
![GitHub repo size](https://img.shields.io/github/repo-size/coder-red/credit-risk-model)
![GitHub last commit](https://img.shields.io/github/last-commit/coder-red/credit-risk-model)
![Type of ML](https://img.shields.io/badge/Type%20of%20ML-Binary%20Classification-red)


# Key findings: Borrowers asking for a higher loan amount and those having existing debt burdens were significantly more likely to default. 

## Author

- [@coder-red](https://www.github.com/coder-red)

## Table of Contents

  - [Borrowers asking for a higher loan amount and those having existing debt burdens were significantly more likely to default](#Borrowers-asking-for-a-higher-loan-amount-and-those-having-existing-debt-burdens-were-significantly-more-likely-to-default)

  - [Author](#author)
  - [Table of Contents](#table-of-contents)
  - [Business problem](#business-problem)
  - [Data source](#data-source)
  - [Methods](#methods)
  - [Tech Stack](#tech-stack)
  - [Quick glance at the results](#quick-glance-at-the-results)
  - [Lessons learned and recommendation](#lessons-learned-and-recommendation)
  - [Limitation and what can be improved](#limitation-and-what-can-be-improved)
  - [Run Locally](#run-locally)
  - [Explore the notebook](#explore-the-notebook)
  - [Repository structure](#repository-structure)


## Business problem

This model predicts if a borrower will pay back a loan or not. Lending institutions use this to make informed decisions on whether to approve a loan or not, interest rate to charge and risk management. 

## Data source

- [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)


## Methods

- Data cleaning,preprocessing and Feature engineering to create predictive variables
- Exploratory data analysis
- Model training and evaluation with Logistic Regression, XGBoost, and LightGBM
- Optuna Hyperparameter tuning 
- SHAP and LIME Model explainability

## Tech Stack

- Python (refer to requirement.txt for the packages used in this project)
- Duckdb (aggregating and joining multiple csvs)
- Scikit-learn, XGBoost, LightGBM (machine learning )
- SHAP & LIME (model explainability)
- Optuna (Hyperparameter tuning) 






Python – for data processing, modeling, and analysis (see requirements.txt for exact packages)



Pandas & NumPy – for data manipulation

Matplotlib & Seaborn – for data visualization and exploratory analysis

– for model explainability

 – for saving and versioning models