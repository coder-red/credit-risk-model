import pandas as pd
import numpy as np
# Import necessary libraries for modelling
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    accuracy_score,
    confusion_matrix
)
import optuna


def calculate_iv(df, feature, target):
    """
    Calculates Information Value (IV) for each features
    Returns IV as a float.
    """
    # Fill NaNs with string for categorical treatment
    df[feature] = df[feature].fillna("NULL")
    
    # Group by feature values
    grouped = df.groupby(feature)[target].agg(['count', 'sum'])
    grouped.rename(columns={'count': 'All', 'sum': 'Bad'}, inplace=True)
    grouped['Good'] = grouped['All'] - grouped['Bad']
    
    # Avoid division by zero
    grouped['Distr_Good'] = (grouped['Good'] / grouped['Good'].sum()).replace(0, 0.0001)
    grouped['Distr_Bad'] = (grouped['Bad'] / grouped['Bad'].sum()).replace(0, 0.0001)
    
    # Calculate WoE and IV
    """
    WoE (Weight of Evidence) is a way to transform categorical features into a number that measures 
        how predictive a category is.

        the formula for WoE is:
        WoE = ln(Distr_Bad / Distr_Good)
    """
    grouped['WoE'] = np.log(grouped['Distr_Bad'] / grouped['Distr_Good'])
    grouped['IV'] = (grouped['Distr_Bad'] - grouped['Distr_Good']) * grouped['WoE']
    
    return grouped['IV'].sum()

def select_features_by_iv(X, y, threshold=0.02, bins=10, verbose=True):
    """
    Selects features based on IV threshold.
    
    Parameters:
    - X: pd.DataFrame, features
    - y: pd.Series, binary target
    - threshold: float, minimum IV to keep feature
    - bins: int, number of bins for numeric features

    Binning is splitting a continuous numeric variable into intervals. 
    Each interval is called a bin
    """
    iv_values = {}
    temp_df = X.copy()
    temp_df['target'] = y
    
    # Bin numeric columns first
    numeric_cols = X.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if X[col].nunique() > bins:
            temp_df[col] = pd.qcut(temp_df[col], q=bins, duplicates='drop').astype(str)
    
    if verbose:
        print(f"Calculating IV for {X.shape[1]} features...")
    
    # Compute IV for all features
    for col in X.columns:
        try:
            iv = calculate_iv(temp_df, col, 'target')
            iv_values[col] = iv
        except Exception as e:
            if verbose:
                print(f"Skipping column {col} due to error: {e}")
    
    # Filter features above threshold
    selected_feats = [col for col, iv in iv_values.items() if iv >= threshold]
    
    if verbose:
        print(f"Selected {len(selected_feats)} features out of {X.shape[1]} (threshold={threshold})")
    
    return selected_feats


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
    """Hyperparameter tuning for XGBoost using Optuna (correct for XGBoost 3.x)."""

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = (n_neg / n_pos) * 2

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "eval_metric": "auc",
            "tree_method": "hist",

            # ✔ correct: early stopping inside constructor (XGB 3.x)
            "early_stopping_rounds": 50,
        }

        model = XGBClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)

    best_params = study.best_params

    # ❗ FIX: put early stopping back for final training
    best_params["early_stopping_rounds"] = 50
    best_params["eval_metric"] = "auc"
    best_params["tree_method"] = "hist"
    best_params["random_state"] = 42

    xgb_tuned = XGBClassifier(**best_params)

    xgb_tuned.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return xgb_tuned



def hyperparameter_tuning_lightgbm(X_train, y_train, X_val, y_val):
    """Hyperparameter tuning for LightGBM using Optuna."""
    
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = (n_neg / n_pos) * 2  # Multiply by 2-3 for better balance

    def objective_lgbm(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", -1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 900),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary",
            "metric": "auc",
            "random_state": 42
        }

        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    study_lgbm = optuna.create_study(direction="maximize")
    study_lgbm.optimize(objective_lgbm, n_trials=40)

    best_params_lgbm = study_lgbm.best_params

    lgbm_tuned = LGBMClassifier(**best_params_lgbm)
    lgbm_tuned.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    return lgbm_tuned




def compare_tuned_and_baseline_models(X_val, y_val, xgb_model, lgbm_model, xgb_tuned, lgbm_tuned):
    """Compare baseline and tuned models side by side."""
    
    def ks_statistic(y_true, y_proba):
        """Kolmogorov–Smirnov statistic"""
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        return max(tpr - fpr)
    
    models = {
        "XGBoost Baseline": xgb_model,
        "XGBoost Tuned": xgb_tuned,
        "LightGBM Baseline": lgbm_model,
        "LightGBM Tuned": lgbm_tuned
    }
    
    results = []
    
    for model_name, model in models.items():
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.15).astype(int)
        
        # Metrics
        roc_auc = roc_auc_score(y_val, y_proba)
        precision_arr, recall_arr, _ = precision_recall_curve(y_val, y_proba)
        pr_auc = auc(recall_arr, precision_arr)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        ks = ks_statistic(y_val, y_proba)
        gini = 2 * roc_auc - 1
        
        results.append({
            "Model": model_name,
            "AUC-ROC": roc_auc,
            "AUC-PR": pr_auc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
            "KS": ks,
            "Gini": gini
        })
    
    df = pd.DataFrame(results)
    return df