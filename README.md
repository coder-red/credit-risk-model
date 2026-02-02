# Credit Risk Prediction with Explainable AI

An end-to-end machine learning system that predicts loan default risk using the Home Credit dataset, combining LightGBM modeling, SHAP/LIME explainability, and regulatory-aligned reporting.


<p align="center">
  <img src="assets/Credit_img.png" alt="Project Banner" width="100%">
</p>


![Python version](https://img.shields.io/badge/Python%20version-3.10%2B-lightgrey)
![GitHub repo size](https://img.shields.io/github/repo-size/coder-red/credit-risk-model)
![GitHub last commit](https://img.shields.io/github/last-commit/coder-red/credit-risk-model)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Type of ML](https://img.shields.io/badge/Type%20of%20ML-Binary%20Classification-red)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/mhmdxch/credit-hf)
![SHAP](https://img.shields.io/badge/SHAP-000000?style=for-the-badge&logo=python&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-272046?style=for-the-badge&logo=pinecone&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)



## Key findings: Borrowers asking for a higher loan amount and those having existing debt burdens were significantly more likely to default. 

## Author

- [@coder-red](https://www.github.com/coder-red)

## Table of Contents
  - [Key findings](#Key-findings)
  - [Author](#author)
  - [Table of Contents](#table-of-contents)
  - [Business Context](#business-context)
  - [Data source](#data-source)
  - [Methods](#methods)
  - [Live Demo](#live-demo)
  - [Tech Stack](#tech-stack)
  - [Quick glance at the results](#quick-glance-at-the-results)
  - [Lessons learned and recommendation](#lessons-learned-and-recommendation)
  - [Limitation and what can be improved](#limitation-and-what-can-be-improved)
  - [Repository structure](#repository-structure)


## Business Context

This model predicts whether a loan applicant will repay or default using data from 250k+ applications. The predictions are made with LightGBM and explained with SHAP and LIME for transparency and outputs are structured in line with EBA compliance standards.


## Data source

- [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)


## Methods

- **Feature Engineering:** Cleaned and Merged data from 8 data sources, then engineered 58 predictive features.
- **Automated Optimization:** Optuna was used for hyperparameter tuning and benchmarking Logistic Regression against XGBoost and LightGBM to find the best model
- **Explainable AI (XAI):** Integrated SHAP and LIME to point out major financial drivers
- **Regulatory RAG:** Built a RAG pipeline to reference model outputs with the EBA (European Banking Authority) standards.


## Live Demo

**[Try the Interactive App](https://huggingface.co/spaces/mhmdxch/credit-hf)**

The trained LightGBM model is deployed as an interactive web application hosted on Hugging Face Spaces using Streamlit where you can:
- Generate random credit risk predictions
- Upload CSV files for batch predictions
- View model performance metrics

<picture>
  <source srcset="assets/credit.gif" type="image/gif">
  <img src="assets/capture.png" alt="App Demo Dashboard showing Credit Risk metrics">
</picture>

## Tech Stack

- Python (refer to requirement.txt for the packages used in this project)
- Duckdb (aggregating and joining multiple csvs)
- Scikit-learn, XGBoost, LightGBM (machine learning )
- SHAP & LIME (model explainability)
- Optuna (Hyperparameter tuning) 
- LangChain (for RAG explanations)
- Pinecone (vector Database)
- Streamlit (interactive web application & deployment)
- Hugging Face(Model deployment)

## Quick glance at the results

Target distribution between the features.

![Bar chart](assets/target_dist.png)

Summary bar of major features

![Bar chart](assets/shap_summary_bar.png)

Confusion matrix of LightGBM.

![Confusion matrix](assets/confusion_matrix_lgbm_tuned.png)

ROC curve of LightGBM.

![ROC curve](assets/roc_curve.png)

Top 3 models 

| Model     	         |    AUC-ROC score     |
|----------------------|----------------------|
| LightGBM(tuned)      | 72.53% 	            |
| XGboost  (tuned)     | 72.56% 	            |
| LightGBM(Baseline)   | 72.08% 	            |


- ***The final model used is: LightGBM because it maximizes recall, catching a larger fraction of potential defaults.***

- ***Metrics used: Recall, AUC-ROC, AUC-PR, Precision,	F1-score, KS, Gini***


## Model Evaluation Strategy

**Primary Metric: ROC-AUC**
Credit risk data is very imbalanced, so ROC-AUC is best here as it measures how well the model does in separating defaulters from non defaulters

**Supporting Metrics: Precision, Recall, F1**
- **Recall** is critical as missing a high-risk borrower leads to real financial loss.
- **Precision** helps ensure we don’t wrongly reject too many good borrowers
- **F1** balances both precision and recall.


## Lessons Learned and Recommendation 

**What I found:**
- Based on the analysis in this project it was found that loan amount, existing debt ratio, and age were the strongest predictors of default
- Hyperparameter tuning barely helped improve the model performance. This suggests that features matter more than tuning

- For imbalanced data, AUC-ROC matters way more than accuracy, and the 0.5 threshold doesn't hold up. For example for Logistic Regression, the optimal threshold was 0.121

**Recommendations:**
- Recommendation would be to focus more on the loan amount when deciding since they carry the most risk and also accept that precision will be low, you'll reject some good customers to catch defaults

## Limitation and What Can Be Improved
- Low precision means 80% of rejected applicants are false positives (lost customers)
- Hyperparameter tuning with Optuna takes 1+ hours
- Incorporate additional external data sources 
- Monitor model performance over time and retrain quarterly

## Repository structure

<details>
  <summary><strong>Repository Structure (click to expand)</strong></summary>

```text

credit-risk-model/
├── assets/                          # Images used in the README 
│   ├── confusion_matrix_lgbm_tuned.png
│   ├── credit.gif
│   ├── roc_curve.png
│   ├── shap_summary_bar.png
│   └── target_dist.png
│
├── data/                            # All data (raw, processed, samples)
│   ├── data_sample/                 # small samples for quick loading 
│   │   ├── application_test_sample.csv
│   │   ├── application_train_sample.csv
│   │   ├── bureau_balance_sample.csv
│   │   ├── bureau_sample.csv
│   │   ├── credit_card_balance_sample.csv
│   │   ├── installments_payments_sample.csv
│   │   ├── POS_CASH_balance_sample.csv
│   │   └── previous_application_sample.csv
│   │
│   ├── processed/                   # cleaned + feature engineered datasets (not tracked in git)
│   │   ├── agg_main.csv
│   │   ├── cleaned_train.csv
│   │   ├── cleaned_val.csv
│   │   ├── feature_engineered_val.csv
│   │   ├── feature_engineered.csv
│   │   ├── target_train.csv
│   │   └── target_val.csv
│   │
│   └── raw/                         # original home credit datasets (not tracked in git)
│       ├── application_test.csv
│       ├── application_train.csv
│       ├── bureau_balance.csv
│       ├── bureau.csv
│       ├── credit_card_balance.csv
│       ├── installments_payments.csv
│       ├── POS_CASH_balance.csv
│       ├── previous_application.csv
│       └── README.md                # Download instructions
│
├── models/                          # Saved trained models (not tracked in git)
│   ├── LightGBM.joblib
│   ├── log_reg.joblib
│   └── XGBoost.joblib
│
├── notebooks/                       # Jupyter notebooks for analysis + modelling + interpretation
│   ├── 01_eda.ipynb
│   ├── 02_modelling.ipynb
│   └── 03_explainability.ipynb
│
├── results/                         # Generated plots and outputs
│   ├── EDA/                         # EDA visualisations
│   │   ├── CODE_GENDER_target_relationship.png
│   │   ├── CODE_GENDER_value_counts.png
│   │   ├── correlation_matrix.png
│   │   ├── missing_value_map.png
│   │   ├── NAME_CONTRACT_TYPE_target_relationship.png
│   │   ├── NAME_CONTRACT_TYPE_value_counts.png
│   │   ├── numeric_boxplots.png
│   │   ├── numeric_histograms.png
│   │   ├── OCCUPATION_TYPE_target_relationship.png
│   │   ├── OCCUPATION_TYPE_value_counts.png
│   │   └── target_dist.png
│   │
│   └── explainability/              # Model interpretation outputs
│       ├── lime_0.png
│       ├── lime_1.png
│       ├── lime_2.png
│       ├── roc_curve.png
│       ├── shap_dependence_age_years.png
│       ├── shap_dependence_avg_debt_ratio.png
│       ├── shap_dependence_CODE_GENDER.png
│       ├── shap_dependence_total_credit_requested.png
│       ├── shap_dependence_value_of_goods_financed.png
│       ├── shap_summary_bar.png
│       └── shap_summary.png
│
├── src/                             # Python modules
│   ├── credit_risk_model/           # Package folder for imports
│   │   ├── __init__.py
│   │   ├── aggregations.py          # Aggregations
│   │   ├── config.py                # Paths and constants
│   │   ├── data_cleaning.py         # Cleaning + preprocessing logic
│   │   ├── data_ingestion.py        # DuckDB ingestion + joins
│   │   ├── feat_eng.py              # Feature engineering functions
│   │   └── model.py                 # Training + evaluation
│   │
│
├── .gitignore                       # Files/folders ignored by git
├── home_credit.duckdb               # DuckDB database file
├── pyproject.toml                   # Build system config 
├── README.md                        # Project overview
└── requirements.txt                 # Required python packages

