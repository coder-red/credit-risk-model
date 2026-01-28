import pandas as pd
import numpy as np


def winsorize_features(df, columns=None):
    """
    Winsorize specified columns to handle outliers.
    Winsorizing replaces the extreme values (outliers) with the nearest "safe" or "normal" values.
    Uses IQR method: clips values beyond Q1 - 1.5*IQR and Q3 + 1.5*IQR
    
    df: DataFrame
    columns: list of column names to winsorize (if None, winsorize all numeric columns)
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound) # .clip makes sure values dont exceed "bounds"
    
    return df


def engineer_features(df):
    """Create new features based on domain knowledge."""

    df = df.copy()

    # We use .replace(0, np.nan) on denominators to avoid "inf" and get "NaN" instead
    df['debt_to_income'] = df['total_debt'] / df['total_income'].replace(0, np.nan)
    df['debt_to_credit'] = df['total_debt'] / df['total_credit_requested'].replace(0, np.nan)
    df['late_payment_ratio'] = df['installments_n_late_payments'] / df['n_prev_apps'].replace(0, np.nan)
    df['income_per_loan'] = df['total_income'] / df['n_loans'].replace(0, np.nan)
    df['income_to_annuity'] = df['total_income'] / df['monthly_loan_payment'].replace(0, np.nan)
    df['bureau_activity_ratio'] = df['n_active_loans'] / df['n_loans'].replace(0, np.nan)

    # Winsorize continuous features (This happens BEFORE median imputation)
    continuous_features = [
        'total_income', 'total_debt', 'total_credit_requested', 
        'monthly_loan_payment', 'debt_to_income', 'debt_to_credit',
        'late_payment_ratio', 'income_per_loan', 'income_to_annuity',
        'avg_utilization', 'n_loans', 'n_active_loans'
    ]
    df = winsorize_features(df, columns=continuous_features)

    # Flags
    df['has_many_active_loans'] = (df['n_active_loans'] > 2).astype(int)
    df['had_any_late'] = (df['total_late_months'].fillna(0) > 0).astype(int)
    df['has_very_late_bureau_loan'] = (df['max_late_single_loan'] > 5).astype(int)
    df['high_credit_utilization'] = (df['avg_utilization'] > 0.8).astype(int)

    return df


