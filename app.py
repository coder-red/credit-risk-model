import joblib
import pandas as pd
import streamlit as st
import numpy as np
from credit_risk_model.config import MODELS_DIR, DATA_PROCESSED
import os

# Import RAG (only if available)
try:
    from retrieval import query_eba_guide
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

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

# Feature to EBA indicator mapping
FEATURE_TO_EBA = {
    'avg_utilization': 'Credit utilization (AQT - Asset Quality)',
    'total_late_months': 'Payment history (AQT - Asset Quality)',
    'debt_to_income': 'Debt-to-income ratio (PFT - Profitability)',
    'n_active_loans': 'Active loans count (CON - Concentration)',
    'credit_income_ratio': 'Credit-to-income ratio (AQT)',
    'age_years': 'Borrower age (demographic risk factor)',
    'total_income': 'Income level (creditworthiness indicator)',
}

# ==================== HEADER ====================
st.title("🤑 Credit Risk Prediction System")
st.markdown("""
This ML-powered system predicts credit default probability using **LightGBM** trained on 250k+ loan applications.
Upload a CSV file or use the demo to see predictions.
""")

# ==================== TABS ====================
if RAG_AVAILABLE:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Demo Prediction", "📁 Batch Upload", "ℹ️ Model Info", "🤖 EBA Guide"])
else:
    tab1, tab2, tab3 = st.tabs(["📊 Demo Prediction", "📁 Batch Upload", "ℹ️ Model Info"])

# ==================== TAB 1: DEMO ====================
with tab1:
    st.subheader("Quick Demo")
    st.info("📌 Click the button below to run a prediction with sample data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎲 Generate Random Prediction", use_container_width=True, type="primary"):
            # Generate random realistic applicant data
            sample_data = {feat: 0.0 for feat in feature_list}
            
            # Randomize key features
            income = np.random.choice([60000, 80000, 120000, 180000, 250000])
            age = np.random.randint(25, 65)
            n_loans = np.random.randint(1, 8)
            n_active = np.random.randint(0, min(n_loans + 1, 5))
            credit_util = np.random.uniform(0.1, 0.8)
            debt_ratio = np.random.uniform(0.5, 5.0)
            late_payments = np.random.choice([0, 0, 0, 1, 2, 5])
            
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
            
            # Show applicant profile
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
            
            # RAG-powered explanation
            if RAG_AVAILABLE and prob > 0.3:
                st.markdown("---")
                st.subheader("🤖 EBA Regulatory Context")
                
                with st.spinner("Analyzing risk factors using EBA guidelines..."):
                    # Identify top risk factors
                    risk_features = []
                    if credit_util > 0.6:
                        risk_features.append(f"high credit utilization ({credit_util:.1%})")
                    if late_payments > 0:
                        risk_features.append(f"{late_payments} late payments")
                    if debt_ratio > 3.0:
                        risk_features.append(f"high debt-to-income ratio ({debt_ratio:.1f})")
                    
                    if risk_features:
                        risk_text = ", ".join(risk_features)
                        query = f"According to EBA guidelines, what are the risks associated with {risk_text} in credit risk assessment?"
                        
                        try:
                            result = query_eba_guide(query)
                            
                            st.info("**Why was this applicant flagged as risky?**")
                            st.write(result["answer"])
                            
                            with st.expander("📚 EBA Sources"):
                                for i, src in enumerate(result["sources"], 1):
                                    st.caption(f"{i}. Page {src['page']}: {src['text']}...")
                        except Exception as e:
                            st.warning("EBA context unavailable (check API keys)")
    
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
            
            with st.expander("Preview Data"):
                st.dataframe(df_upload.head(10))
            
            if st.button("⚡ Run Predictions", type="primary"):
                with st.spinner("Processing..."):
                    df_features = df_upload[feature_list]
                    probabilities = model.predict_proba(df_features)[:, 1]
                    
                    df_upload['Default_Probability'] = probabilities
                    df_upload['Risk_Category'] = pd.cut(
                        probabilities,
                        bins=[0, 0.2, 0.4, 1.0],
                        labels=['Low Risk', 'Medium Risk', 'High Risk']
                    )
                    
                    st.dataframe(df_upload)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total", len(df_upload))
                    col2.metric("High Risk", sum(probabilities >= 0.4))
                    col3.metric("Avg Risk", f"{probabilities.mean():.1%}")
                    
                    csv = df_upload.to_csv(index=False)
                    st.download_button(
                        "💾 Download Results",
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
    
    # EBA feature mapping
    if RAG_AVAILABLE:
        st.subheader("🏛️ EBA Indicator Mapping")
        st.markdown("Our model features align with EBA risk indicators:")
        
        for feature, eba_indicator in list(FEATURE_TO_EBA.items())[:7]:
            st.text(f"• {feature} → {eba_indicator}")
    
    st.info("💡 **Note**: This is a demonstration project. Real credit decisions should involve additional factors and human oversight.")

# ==================== TAB 4: EBA GUIDE CHATBOT ====================
if RAG_AVAILABLE:
    with tab4:
        st.subheader("🤖 EBA Methodological Guide Assistant")
        st.markdown("""
        Ask questions about **EBA credit risk guidelines** and how they relate to this model.
        """)
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("📚 Sources"):
                        for i, src in enumerate(message["sources"], 1):
                            st.caption(f"{i}. Page {src['page']}")
        
        # Example questions
        st.markdown("**💡 Example questions:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("What are AQT indicators?"):
                st.session_state.example_query = "What are the main asset quality (AQT) indicators in the EBA guide?"
            if st.button("Explain NPE ratio"):
                st.session_state.example_query = "How should non-performing exposure (NPE) ratio be calculated?"
        
        with col2:
            if st.button("Credit utilization risk"):
                st.session_state.example_query = "Why is high credit utilization a risk factor according to EBA guidelines?"
            if st.button("Late payment impact"):
                st.session_state.example_query = "What is the impact of late payments on credit risk per EBA standards?"
        
        # Chat input
        if prompt := st.chat_input("Ask about EBA guidelines..."):
            query_text = prompt
        elif "example_query" in st.session_state:
            query_text = st.session_state.example_query
            del st.session_state.example_query
        else:
            query_text = None
        
        if query_text:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query_text})
            with st.chat_message("user"):
                st.markdown(query_text)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching EBA guide..."):
                    try:
                        result = query_eba_guide(query_text)
                        
                        st.markdown(result['answer'])
                        
                        with st.expander("📚 Sources"):
                            for i, src in enumerate(result['sources'], 1):
                                st.caption(f"{i}. Page {src['page']}")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result['answer'],
                            "sources": result['sources']
                        })
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Make sure you've run ingestion.py and set API keys in .env")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("assets/Credit_img.png", use_container_width=True)
    
    st.markdown("### 💻 Portfolio Project")
    st.markdown("""
    **Built by:** Ahmed Mohammed 
    
    **GitHub:** https://github.com/coder-red 
    
    **LinkedIn:** https://www.linkedin.com/in/coder-red
    """)
    
    st.divider()
    
    if RAG_AVAILABLE:
        st.success("✅ RAG System Active")
        st.caption("Powered by EBA Guide + Groq + Pinecone")
    else:
        st.warning("⚠️ RAG Unavailable")
        st.caption("Run ingestion.py to enable")