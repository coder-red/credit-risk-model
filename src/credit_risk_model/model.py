import pandas as pd
import numpy as np
# Import necessary libraries for modelling
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    accuracy_score,
    confusion_matrix
)
import optuna


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train a Logistic Regression model."""
     # Detect binary columns
    binary_cols = [col for col in X_train.columns if X_train[col].nunique() <= 3] # including 0,1 and 3 for gender col
    continuous_cols = [col for col in X_train.columns if col not in binary_cols]


    # use pipeline to scale as it is needed for logistic regression
    preprocessor = ColumnTransformer([
        ('scale', StandardScaler(), continuous_cols),
        ('pass', 'passthrough', binary_cols)
    ])
    log_reg_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('logistic_reg', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])


    # train
    log_reg_pipe.fit(X_train, y_train)

    # predict # remove these????  
    y_pred = log_reg_pipe.predict(X_val)
    y_proba = log_reg_pipe.predict_proba(X_val)[:, 1]

    return log_reg_pipe, y_proba, y_pred



def train_xgboost(X_train, y_train, X_val, y_val):
    """Train an XGBoost model."""
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='aucpr'  # Better for imbalanced data
    )

    # Fit model
    xgb_model.fit(X_train, y_train)

    # Predict
    y_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
    y_pred_xgb = (y_proba_xgb >= 0.15).astype(int)

    return xgb_model, y_proba_xgb, y_pred_xgb


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train a LightGBM model."""
    lgbm_model = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )

    lgbm_model.fit(X_train, y_train)

    # Predict
    y_proba_lgbm = lgbm_model.predict_proba(X_val)[:, 1]
    y_pred_lgbm = (y_proba_lgbm >= 0.15).astype(int)


    return lgbm_model, y_proba_lgbm, y_pred_lgbm


def compare_models(X_val, y_val, log_reg_pipe, xgb_model, lgbm_model):
    def ks_statistic(y_true, y_proba):
        """Kolmogorov–Smirnov statistic (credit risk classic)"""
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        return max(tpr - fpr)

    models = {
        "LogisticRegression": log_reg_pipe,
        "XGBoost": xgb_model,
        "LightGBM": lgbm_model
    }

    results = {}

    for model_name, model in models.items():

        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.15).astype(int)  # decision threshold

        # Core metrics
        roc = roc_auc_score(y_val, y_proba)
        precision_arr, recall_arr, _ = precision_recall_curve(y_val, y_proba)
        pr_auc = auc(recall_arr, precision_arr)

        # Classification metrics
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        # Credit risk metrics
        ks = ks_statistic(y_val, y_proba)
        gini = 2 * roc - 1

        results[model_name] = {
            "AUC-ROC": roc,
            "AUC-PR": pr_auc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
            "KS": ks,
            "Gini": gini
        }

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values(by='AUC-ROC', ascending=False)


    return results_df


def hyperparameter_tuning_xgboost(X_train, y_train, X_val, y_val):
    """Hyperparameter tuning for XGBoost using Optuna."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
            "eval_metric": "auc",
            "tree_method": "hist"   
        }

        model = XGBClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)  

    # Get best parameters
    best_params = study.best_params
    best_params

    # Train final model with best parameters
    xgb_tuned = XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric="auc",
        tree_method="hist"
    )

    xgb_tuned.fit(X_train, y_train)

    y_proba = xgb_tuned.predict_proba(X_val)[:, 1]
    roc_auc_score(y_val, y_proba)

    return xgb_tuned



def hyperparameter_tuning_lightgbm(X_train, y_train, X_val, y_val):
    """Hyperparameter tuning for LightGBM using Optuna."""

    def objective_lgbm(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "random_state": 42,
            "objective": "binary",
            "metric": "auc"
        }

        model = LGBMClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
        )

        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)


    # Run Optuna study
    study_lgbm = optuna.create_study(direction="maximize")
    study_lgbm.optimize(objective_lgbm, n_trials=40)


    # Get best parameters
    best_params_lgbm = study_lgbm.best_params


    # Train final model with best parameters
    lgbm_tuned = LGBMClassifier(
        **best_params_lgbm,
        objective="binary",
        metric="auc",
        random_state=42
    )

    lgbm_tuned.fit(X_train, y_train)

    return lgbm_tuned, y_val, X_val


def compare_tuned_and_baseline_models(X_val, y_val, log_reg_pipe,xgb_model,lgbm_model, xgb_tuned, lgbm_tuned):
    """Compare baseline and tuned models."""

        # Unpack if tuples
    if isinstance(xgb_tuned, tuple):
        xgb_tuned = xgb_tuned[0]
    if isinstance(lgbm_tuned, tuple):
        lgbm_tuned = lgbm_tuned[0]


    results = {
        'Model': ['LogisticRegression', 'XGBoost', 'LightGBM'],
        'Baseline AUC': [
            roc_auc_score(y_val, log_reg_pipe.predict_proba(X_val)[:, 1]),
            roc_auc_score(y_val, xgb_model.predict_proba(X_val)[:, 1]),
            roc_auc_score(y_val, lgbm_model.predict_proba(X_val)[:, 1])
        ],
        'Tuned AUC': [
            None,
            roc_auc_score(y_val, xgb_tuned.predict_proba(X_val)[:, 1]),
            roc_auc_score(y_val, lgbm_tuned.predict_proba(X_val)[:, 1])
        ]
    }

    df = pd.DataFrame(results)

    return df