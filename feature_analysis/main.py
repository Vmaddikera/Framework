import streamlit as st
import vif_lime_streamlit
import target_correlations_streamlit
import strategy_analysis_streamlit
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    st.set_page_config(
        page_title="Trading Strategy Analysis Dashboard",
        layout="wide"
    )

    st.title("Trading Strategy Analysis Dashboard")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="main_uploader")
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Shape: {df.shape}")
            st.info(f"Columns: {list(df.columns)}")
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            df = None

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Strategy Analysis",
        "VIF & LIME Analysis",
        "Target Correlations",
        "SHAP Analysis",
        "Optimal Trading Weights"
    ])

    with tab1:
        try:
            strategy_analysis_streamlit.main(df)
        except Exception as e:
            st.error(f"Error in Strategy Analysis: {str(e)}")
            st.info("Please check your data format and try again.")
    
    with tab2:
        try:
            if df is not None:
                st.info("Starting VIF & LIME Analysis...")
                vif_lime_streamlit.main(df)
            else:
                st.warning("Please upload a CSV file first to run VIF & LIME Analysis.")
        except Exception as e:
            st.error(f"Error in VIF & LIME Analysis: {str(e)}")
            st.info("Please check your data format and try again.")
        
    with tab3:
        try:
            if df is not None:
                st.info("Starting Target Correlations Analysis...")
                target_correlations_streamlit.main(df)
            else:
                st.warning("Please upload a CSV file first to run Target Correlations Analysis.")
        except Exception as e:
            st.error(f"Error in Target Correlations Analysis: {str(e)}")
            st.info("Please check your data format and try again.")
    
    with tab4:
        try:
            if df is not None:
                st.info("Starting SHAP Analysis...")
                import shap_analysis
                shap_analysis.shap_analysis_y_actual_streamlit(df)
            else:
                st.warning("Please upload a CSV file first to run SHAP Analysis.")
        except Exception as e:
            st.error(f"Error in SHAP Analysis: {str(e)}")
            st.info("Please check your data format and try again.")
    
    with tab5:
        try:
            if df is not None:
                st.info("Starting Optimal Trading Weights Analysis...")
                import feature_weighting
                
                # Use simplified optimal trading weights interface
                weighting_system = feature_weighting.FeatureWeightingSystem()
                optimal_weights = weighting_system.display_simple_optimal_trading_interface(df)
                
            else:
                st.warning("Please upload a CSV file first to run Optimal Trading Weights Analysis.")
        except Exception as e:
            st.error(f"Error in Optimal Trading Weights Analysis: {str(e)}")
            st.info("Please check your data format and try again.")
    
if __name__ == "__main__":
    main() 