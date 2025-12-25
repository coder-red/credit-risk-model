import joblib
import pandas as pd
from credit_risk_model.config import MODELS_DIR, DATA_PROCESSED


model = joblib.load(f"{MODELS_DIR}/LightGBM.joblib")
feature_list = joblib.load(f"{DATA_PROCESSED}/feature_list.joblib")
# user input
user_input = {feat: 0.0 for feat in feature_list}
df = pd.DataFrame([user_input])

# enforce correct order
df = df[feature_list]

prob = model.predict_proba(df)[0, 1]

print(df, prob)
