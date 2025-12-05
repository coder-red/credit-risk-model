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


## Quick glance at the results

Target distribution between the features.

![Bar chart](assets\target_dist.png)

Summary bar of major features

![Bar chart](assets/shap_summary_bar.png)

Confusion matrix of LightGBM.

![Confusion matrix](assets/confusion_matrix_lgbm_tuned.png)

ROC curve of LightGBM.

![ROC curve](assets/roc_curve.png)

Top 3 models (with default parameters)

| Model     	         |    AUC-ROC score     |
|----------------------|----------------------|
| LightGBM(tuned)      | 72.57% 	            |
| XGboost  (tuned)     | 72.42% 	            |
| Logistic Regression  | 69.80% 	            |


- ***The final model used is: XGboost***
- ***Metrics used: Recall, AUC-ROC, AUC-PR, Precision,	F1-score, KS, Gini***


### Model Evaluation Strategy

**Primary Metric: ROC-AUC**
Credit risk data is very imbalanced, so ROC-AUC is best here as it measures how well the model does in separating defaulters from non defaulters.

**Supporting Metrics: Precision, Recall, F1**
- **Recall** is critical as missing a high-risk borrower leads to real financial loss.
- **Precision** helps ensure we don’t wrongly reject too many good borrowers.
- **F1** balances both.

This combination gives a realistic view of how the model performs in a real lending environment.



