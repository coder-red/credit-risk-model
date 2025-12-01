import duckdb
import pandas as pd
from sklearn.preprocessing import LabelEncoder



def fill_missing_values(df):
    """Fill missing numeric with median, categorical with mode."""
    df = df.copy()

    # numeric columns
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    return df


def encode_contract_type(df):
    """Binary encode NAME_CONTRACT_TYPE."""
    df = df.copy()
    df['NAME_CONTRACT_TYPE'] = df['NAME_CONTRACT_TYPE'].map({'Cash loans': 0, 'Revolving loans': 1})
    return df

def encode_gender(df):
    """Binary encode CODE_GENDER."""
    df = df.copy()
    df['CODE_GENDER'] = df['CODE_GENDER'].map({'M': 0, 'F': 1, 'XNA': 2})
    return df


def encode_occupation_type(df):
    """One-hot encode OCCUPATION_TYPE."""
    df = df.copy()
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Unknown')
    dummies = pd.get_dummies(df['OCCUPATION_TYPE'], drop_first=True).astype(int)
    df = pd.concat([df.drop('OCCUPATION_TYPE', axis=1), dummies], axis=1)
    return df
    
def clean_all(df):
    """Apply all cleaning steps."""
    df = fill_missing_values(df)
    df = encode_contract_type(df)
    df = encode_gender(df)
    df = encode_occupation_type(df)
    return df