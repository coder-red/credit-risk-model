import pandas as pd
import numpy as np

def engineer_features(df):
    """Create new features based on domain knowledge."""
    df = df.copy()
    eps = 1e-6  # (epsilon) small constant to avoid division by zero

    # ratios
    df['debt_to_income'] = df['total_debt'] / (df['total_income'] + eps)
    df['debt_to_credit'] = df['total_debt'] / (df['total_credit_requested'] + eps)
    df['late_payment_ratio'] = df['installments_n_late_payments'] / (df['n_prev_apps'] + eps)
    df['income_per_loan'] = df['total_income'] / (df['n_loans'] + eps)
    df['income_to_annuity'] = df['total_income'] / (df['monthly_loan_payment'] + eps)  # higher is safer

    df['has_many_active_loans'] = (df['n_active_loans'] > 2).astype(int)

    # flags
    df['had_any_late'] = (df['total_late_months'].fillna(0) > 0).astype(int)
    df['has_very_late_bureau_loan'] = (df['max_late_single_loan'] > 5).astype(int)  
    df['high_credit_utilization'] = (df['avg_utilization'] > 0.8).astype(int)
    df['bureau_activity_ratio'] = df['n_active_loans'] / (df['n_loans'] + eps)

    return df


