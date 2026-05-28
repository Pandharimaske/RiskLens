"""
RiskLens Streamlit Dashboard
Purpose: Interactive UI for model predictions, batch processing, and monitoring
Provides single prediction, batch upload, and drift monitoring visualization

Author: RiskLens MLOps
Date: 2026-05-27
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Page configuration
st.set_page_config(
    page_title="RiskLens - Insurance Risk Prediction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_artifacts():
    """Load model and preprocessing artifacts."""
    try:
        model = joblib.load("artifacts/calibrated_model.pkl")
        preprocessor = joblib.load("artifacts/preprocessor.pkl")
        feature_names = joblib.load("artifacts/feature_names.pkl")
        return model, preprocessor, feature_names
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        return None, None, None

@st.cache_data
def load_drift_metrics():
    """Load drift monitoring metrics."""
    try:
        with open("monitoring/drift_metrics.json") as f:
            return json.load(f)
    except:
        return None

def preprocess_input(data, preprocessor, feature_names):
    """Preprocess input data for prediction."""
    try:
        df = pd.DataFrame(data, index=[0])
        df = df[feature_names]
        processed = preprocessor.transform(df)
        return processed
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None

def make_prediction(model, processed_data):
    """Make single or batch predictions."""
    try:
        pred_class = model.predict(processed_data)
        pred_proba = model.predict_proba(processed_data)[:, 1]
        return pred_class, pred_proba
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔍 RiskLens - Insurance Risk Prediction")
        st.markdown("*AI-powered vehicle insurance claim risk assessment*")
    with col2:
        st.metric("Status", "✅ Active", delta="Production Ready")
    
    # Load artifacts
    model, preprocessor, feature_names = load_artifacts()
    if model is None:
        st.error("Could not load model artifacts. Ensure artifacts/ directory exists.")
        return
    
    # Feature definitions
    feature_descriptions = {
        'Age': 'Customer age (years)',
        'Driving_License': 'Has valid driving license (0/1)',
        'Region_Code': 'Geographic region code (1-28)',
        'Previously_Insured': 'Previous insurance history (0/1)',
        'Vehicle_Age_Numeric': 'Vehicle age (years)',
        'Annual_Premium': 'Annual premium amount (₹)',
        'Policy_Sales_Channel': 'Sales channel (1-3)',
        'Vintage': 'Days as customer (days)',
        'Fuel_Type': 'Fuel type (Petrol/Diesel)',
        'Vehicle_Damage': 'Previous vehicle damage (Yes/No)',
        'Premium_Bucket': 'Premium category (Low/Medium/High)',
        'Customer_Tenure_Segment': 'Tenure segment (Months)',
        'Premium_per_Vehicle_Year': 'Normalized premium metric',
        'Age_Bracket': 'Age category',
        'Income_Level': 'Income segment',
        'Risk_Category': 'Risk segment',
        'Loyalty_Score': 'Customer loyalty metric'
    }
    
    # Sidebar - Navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Mode",
        ["🔮 Single Prediction", "📥 Batch Prediction", "📈 Monitoring Dashboard"],
        help="Choose prediction mode or view monitoring dashboard"
    )
    
    st.sidebar.divider()
    st.sidebar.info(
        "**RiskLens MLOps Platform**\n\n"
        "• Phase 9: SHAP Explainability ✅\n"
        "• Phase 10: DVC Pipeline ✅\n"
        "• Phase 11: Drift Monitoring ✅\n\n"
        "Model Version: 0.0400\n"
        "Calibration: Sigmoid (Platt)\n"
        "Framework: LightGBM"
    )
    
    # Tabs for different views
    if page == "🔮 Single Prediction":
        st.header("Single Prediction")
        st.markdown("Enter customer and vehicle details for individual risk assessment")
        
        # Input form
        col1, col2 = st.columns(2)
        
        input_data = {}
        
        # Numeric inputs
        with col1:
            st.subheader("Customer Information")
            input_data['Age'] = st.number_input("Age", min_value=18, max_value=100, value=35)
            input_data['Annual_Premium'] = st.number_input("Annual Premium (₹)", min_value=0, max_value=500000, value=40000)
            input_data['Vintage'] = st.number_input("Days as Customer", min_value=0, max_value=3000, value=365)
            input_data['Previously_Insured'] = st.selectbox("Previously Insured", [0, 1], format_func=lambda x: "Yes" if x else "No")
            
        with col2:
            st.subheader("Vehicle Information")
            input_data['Vehicle_Age_Numeric'] = st.number_input("Vehicle Age (years)", min_value=0, max_value=50, value=5)
            input_data['Driving_License'] = st.selectbox("Has Driving License", [0, 1], format_func=lambda x: "Yes" if x else "No")
            input_data['Region_Code'] = st.number_input("Region Code", min_value=1, max_value=28, value=5)
            input_data['Policy_Sales_Channel'] = st.number_input("Sales Channel", min_value=1, max_value=3, value=1)
        
        # Additional features (with defaults for missing ones)
        other_features = [f for f in feature_names if f not in input_data]
        for feature in other_features:
            if feature not in input_data:
                input_data[feature] = 0  # Default value
        
        # Prediction button
        if st.button("🔮 Predict Risk", type="primary", use_container_width=True):
            # Preprocess
            processed_data = preprocess_input(input_data, preprocessor, feature_names)
            
            if processed_data is not None:
                # Predict
                pred_class, pred_proba = make_prediction(model, processed_data)
                
                if pred_class is not None:
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        risk_label = "🔴 HIGH RISK" if pred_class[0] == 1 else "🟢 LOW RISK"
                        st.metric("Risk Classification", risk_label)
                    
                    with col2:
                        st.metric("Risk Probability", f"{pred_proba[0]:.2%}")
                    
                    with col3:
                        st.metric("Decision Threshold", "0.0400")
                    
                    # Risk gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=pred_proba[0] * 100,
                        title={'text': "Claim Risk %"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 4], 'color': "lightgray"},
                                {'range': [4, 100], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 4
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explanation
                    st.subheader("📊 Risk Assessment")
                    if pred_proba[0] >= 0.04:
                        st.warning("⚠️ This customer has elevated claim risk. Consider premium adjustment or enhanced underwriting review.")
                    else:
                        st.success("✓ This customer poses acceptable risk. Standard underwriting process recommended.")
    
    elif page == "📥 Batch Prediction":
        st.header("Batch Prediction")
        st.markdown("Upload CSV file for batch risk assessment")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload CSV file", type="csv", help="CSV must contain feature columns")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} records")
                
                # Preview
                st.subheader("Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Predict
                if st.button("🚀 Run Batch Predictions", type="primary", use_container_width=True):
                    # Prepare data
                    df_proc = df[feature_names].fillna(0)
                    processed_data = preprocessor.transform(df_proc)
                    
                    # Predict
                    pred_class, pred_proba = make_prediction(model, processed_data)
                    
                    if pred_class is not None:
                        # Add predictions to dataframe
                        results = df.copy()
                        results['risk_class'] = pred_class
                        results['risk_probability'] = pred_proba
                        results['risk_label'] = results['risk_probability'].apply(
                            lambda x: "HIGH RISK" if x >= 0.04 else "LOW RISK"
                        )
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(results[['risk_class', 'risk_probability', 'risk_label']], use_container_width=True)
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Records", len(results))
                        with col2:
                            high_risk = (results['risk_class'] == 1).sum()
                            st.metric("High Risk", f"{high_risk} ({100*high_risk/len(results):.1f}%)")
                        with col3:
                            st.metric("Avg Risk", f"{results['risk_probability'].mean():.2%}")
                        
                        # Distribution plot
                        fig = px.histogram(
                            results,
                            x='risk_probability',
                            nbins=30,
                            title="Risk Probability Distribution",
                            labels={'risk_probability': 'Risk Probability', 'count': 'Count'},
                            color_discrete_sequence=['#636EFA']
                        )
                        fig.add_vline(x=0.04, line_dash="dash", line_color="red", annotation_text="Threshold")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    else:  # Monitoring Dashboard
        st.header("📈 Monitoring Dashboard")
        st.markdown("Real-time model performance and drift monitoring")
        
        # Load drift metrics
        drift_metrics = load_drift_metrics()
        
        if drift_metrics:
            # KPIs
            col1, col2, col3, col4 = st.columns(4)
            
            data_drift = drift_metrics['data_drift']
            pred_drift = drift_metrics['prediction_drift']
            
            with col1:
                st.metric(
                    "Features Analyzed",
                    data_drift['total_features'],
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Features with Drift",
                    data_drift['features_with_drift'],
                    delta=None,
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    "Data Drift Status",
                    "🟢 Stable" if data_drift['features_with_drift'] == 0 else "🔴 Drift Detected",
                    delta=None
                )
            
            with col4:
                st.metric(
                    "Pred. Drift Status",
                    "🟢 Stable" if not pred_drift['has_prediction_drift'] else "🔴 Drift Detected",
                    delta=None
                )
            
            st.divider()
            
            # Feature drift details
            st.subheader("Feature-Level Drift Analysis")
            
            drift_df = pd.DataFrame([
                {
                    'Feature': feature,
                    'Ref Mean': details['reference_mean'],
                    'Curr Mean': details['current_mean'],
                    'Drift Score': details['drift_score'],
                    'Status': '⚠️ Drift' if details['has_drift'] else '✓ OK'
                }
                for feature, details in data_drift['drift_details'].items()
            ]).sort_values('Drift Score', ascending=False)
            
            st.dataframe(
                drift_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Drift Score': st.column_config.NumberColumn(format='%.4f'),
                    'Ref Mean': st.column_config.NumberColumn(format='%.4f'),
                    'Curr Mean': st.column_config.NumberColumn(format='%.4f')
                }
            )
            
            # Prediction distribution
            st.subheader("Prediction Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean Probability", f"{pred_drift['prediction_probability_mean']:.4f}")
                st.metric("Std Deviation", f"{pred_drift['prediction_probability_std']:.4f}")
            
            with col2:
                st.metric("Min Probability", f"{pred_drift['prediction_probability_min']:.4f}")
                st.metric("Max Probability", f"{pred_drift['prediction_probability_max']:.4f}")
            
            st.divider()
            
            # Last updated
            st.caption(f"Last updated: {data_drift.get('timestamp', 'Unknown')}")
        
        else:
            st.warning("Drift metrics not available. Run Phase 11 to generate monitoring data.")

if __name__ == "__main__":
    main()
