import joblib
import pandas as pd
import streamlit as st
import numpy as np
from credit_risk_model.config import MODELS_DIR, DATA_PROCESSED
import os



# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🤑",
    layout="wide"
)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model_and_features():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model = joblib.load(os.path.join(BASE_DIR, "models", "LightGBM.joblib"))
    feature_list = joblib.load(os.path.join(BASE_DIR, "data", "processed", "feature_list.joblib"))
    return model, feature_list

model, feature_list = load_model_and_features()

# ==================== HEADER ====================
st.title(" Credit Risk Prediction System")
st.markdown("""
This ML-powered system predicts credit default probability using **LightGBM** trained on 250k+ loan applications.
Upload a CSV file or use the demo to see predictions.
""")

# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs(["📊 Demo Prediction", "📁 Batch Upload", "ℹ️ Model Info"])

# ==================== TAB 1: DEMO ====================
with tab1:
    st.subheader("Quick Demo")
    st.info(" Click the button below to run a prediction with sample data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button(" Generate Random Prediction", use_container_width=True, type="primary"):
            # Generate random realistic applicant data
            sample_data = {feat: 0.0 for feat in feature_list}
            
            # Randomize key features with realistic distributions
            income = np.random.choice([60000, 80000, 120000, 180000, 250000])
            age = np.random.randint(25, 65)
            n_loans = np.random.randint(1, 8)
            n_active = np.random.randint(0, min(n_loans + 1, 5))
            credit_util = np.random.uniform(0.1, 0.8)
            debt_ratio = np.random.uniform(0.5, 5.0)
            late_payments = np.random.choice([0, 0, 0, 1, 2, 5])  # Most have none
            
            sample_data.update({
                'total_income': float(income),
                'age_years': float(age),
                'n_loans': float(n_loans),
                'n_active_loans': float(n_active),
                'credit_income_ratio': debt_ratio,
                'avg_utilization': credit_util,
                'total_late_months': float(late_payments),
                'debt_to_income': debt_ratio * 0.8,
                'approval_rate': np.random.uniform(0.5, 1.0),
            })
            
            df = pd.DataFrame([sample_data])[feature_list]
            prob = model.predict_proba(df)[0, 1]
            
            # Display result
            st.markdown("---")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Default Probability", f"{prob:.1%}")
            
            with col_b:
                if prob < 0.2:
                    st.success("🟢 LOW RISK")
                elif prob < 0.4:
                    st.warning("🟡 MEDIUM RISK")
                else:
                    st.error("🔴 HIGH RISK")
            
            with col_c:
                st.metric("Risk Score", f"{(1-prob)*100:.0f}/100")
            
            # Show feature importance
            with st.expander("🔍 View Generated Applicant Profile"):
                display_features = {
                    'Annual Income': f'${income:,}',
                    'Age': f'{age} years',
                    'Total Loans': f'{n_loans}',
                    'Active Loans': f'{n_active}',
                    'Credit Utilization': f'{credit_util:.1%}',
                    'Late Payments': f'{late_payments} months',
                    'Total Features': f'{len(feature_list)} analyzed'
                }
                for k, v in display_features.items():
                    st.text(f"• {k}: {v}")
    
    with col2:
        st.markdown("### 📈 Model Performance")
        st.metric("AUC-ROC", "0.73")
        st.metric("Features", len(feature_list))
        st.metric("Training Size", "246k")

# ==================== TAB 2: BATCH UPLOAD ====================
with tab2:
    st.subheader("Batch Predictions")
    st.markdown("Upload a CSV file with applicant data to get predictions for multiple cases.")
    
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=['csv'],
        help=f"CSV must contain these {len(feature_list)} columns: {', '.join(feature_list[:5])}..."
    )
    
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df_upload)} records")
            
            # Show preview
            with st.expander("Preview Data"):
                st.dataframe(df_upload.head(10))
            
            if st.button("⚡ Run Predictions", type="primary"):
                with st.spinner("Processing..."):
                    # Ensure correct columns
                    df_features = df_upload[feature_list]
                    
                    # Predict
                    probabilities = model.predict_proba(df_features)[:, 1]
                    
                    # Add results
                    df_upload['Default_Probability'] = probabilities
                    df_upload['Risk_Category'] = pd.cut(
                        probabilities,
                        bins=[0, 0.2, 0.4, 1.0],
                        labels=['Low Risk', 'Medium Risk', 'High Risk']
                    )
                    
                    # Show results
                    st.dataframe(df_upload)
                    
                    # Summary
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total", len(df_upload))
                    col2.metric("High Risk", sum(probabilities >= 0.4))
                    col3.metric("Avg Risk", f"{probabilities.mean():.1%}")
                    
                    # Download
                    csv = df_upload.to_csv(index=False)
                    st.download_button(
                        " Download Results",
                        csv,
                        "predictions.csv",
                        "text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure your CSV has all required feature columns.")

# ==================== TAB 3: MODEL INFO ====================
with tab3:
    st.subheader("About This Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📄 Model Details
        - **Algorithm**: LightGBM Classifier
        - **Training Data**: 246,008 loan applications
        - **Features**: 60+ financial & behavioral metrics
        - **Performance**: 73% AUC-ROC
        
        ### 📊 Key Features Analyzed
        - Income & debt ratios
        - Credit utilization
        - Payment history
        - Number of active loans
        - Previous application history
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Risk Thresholds
        
        **🟢 Low Risk (0-20%)**  
        Strong creditworthiness, recommend approval
        
        **🟡 Medium Risk (20-40%)**  
        Moderate risk, additional review needed
        
        **🔴 High Risk (40%+)**  
        Elevated default risk, recommend decline
        
        ### 🛠️ Technical Stack
        - Python 
        - LightGBM 
        - Optuna
        - Streamlit
        - Scikit-learn
        """)
    
    st.markdown("---")
    st.info("💡 **Note**: This is a demonstration project. Real credit decisions should involve additional factors and human oversight.")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("assets/Credit_img.png", use_container_width=True)
    
    st.markdown("### 💻 Portfolio Project")
    st.markdown("""


    **Built by:** Ahmed Mohammed 


    **GitHub:**  https://github.com/coder-red 

    **LinkedIn:** https://www.linkedin.com/in/coder-red


    """)
    
    st.divider()
    
 