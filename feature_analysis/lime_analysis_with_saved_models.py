import pandas as pd
import numpy as np
import pickle
import os
import glob
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# Set matplotlib backend for Streamlit
import matplotlib
matplotlib.use('Agg')

class LIMEAnalyzer:
    """
    LIME analyzer for using saved ML models from pickle files.
    Models are loaded every 100 ticks and used for LIME analysis.
    """
    
    def __init__(self, models_dir="model_evaluation/models"):
        self.models_dir = models_dir
        self.model_files = self._get_model_files()
        
    def _get_model_files(self):
        """Get all model pickle files sorted by tick number."""
        pattern = os.path.join(self.models_dir, "model_tick_*.pkl")
        model_files = glob.glob(pattern)
        
        # Sort by tick number
        model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        print(f"Found {len(model_files)} model files")
        return model_files
    
    def load_model_for_tick(self, tick_number):
        """
        Load the appropriate model for a given tick number.
        Models are saved every 100 ticks, so we find the closest model.
        """
        # Find the model file for this tick range
        base_tick = (tick_number // 100) * 100
        
        model_file = os.path.join(self.models_dir, f"model_tick_{base_tick}.pkl")
        
        if not os.path.exists(model_file):
            print(f"Model file not found: {model_file}")
            return None
        
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            print(f"Loaded model for tick {tick_number} from {model_file}")
            return model_data
            
        except Exception as e:
            print(f"Error loading model {model_file}: {e}")
            return None
    
    def extract_model_and_scaler(self, model_data):
        """
        Extract the actual ML model and scaler from the saved model data.
        """
        if isinstance(model_data, dict):
            # Common keys in saved model data
            possible_model_keys = ['model', 'ml_model', 'regressor', 'classifier']
            possible_scaler_keys = ['scaler', 'feature_scaler', 'standard_scaler']
            
            model = None
            scaler = None
            
            # Find model
            for key in possible_model_keys:
                if key in model_data:
                    model = model_data[key]
                    break
            
            # Find scaler
            for key in possible_scaler_keys:
                if key in model_data:
                    scaler = model_data[key]
                    break
            
            return model, scaler
        else:
            # Assume the data is the model itself
            return model_data, None
    
    def perform_lime_analysis(self, df, start_row=2, tick_interval=100, num_samples=3):
        """
        Perform LIME analysis using saved models for every 100 ticks.
        """
        st.header("LIME Analysis with Saved Models")
        
        # Prepare data for analysis
        analysis_chunks = self._prepare_data_for_analysis(df, start_row, tick_interval)
        
        if not analysis_chunks:
            st.error("No analysis chunks prepared. Check model files and data.")
            return
        
        # Select chunks for LIME analysis (to avoid overwhelming output)
        selected_chunks = analysis_chunks[:5]  # First 5 chunks
        
        for chunk_idx, chunk_info in enumerate(selected_chunks):
            st.subheader(f"LIME Analysis - Chunk {chunk_idx+1} (Ticks {chunk_info['tick_number']}-{chunk_info['tick_number']+tick_interval})")
            
            try:
                # Get feature columns
                feature_cols = [col for col in chunk_info['data_chunk'].columns 
                              if col not in ['y_actual', 'y_predict', 'forecast', 'actual_target', 'signal_correct', 'executed_signal']]
                
                X = chunk_info['data_chunk'][feature_cols]
                y_actual = chunk_info['data_chunk']['y_actual']
                
                # Apply scaler if available
                if chunk_info['scaler'] is not None:
                    X_scaled = chunk_info['scaler'].transform(X)
                else:
                    X_scaled = X
                
                # Create LIME explainer
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_scaled,
                    feature_names=feature_cols,
                    class_names=['prediction'],
                    mode='regression'
                )
                
                # Analyze samples
                for sample_idx in range(min(num_samples, len(X_scaled))):
                    st.write(f"**Sample {sample_idx+1} Analysis:**")
                    
                    # Get LIME explanation
                    exp = explainer.explain_instance(
                        X_scaled[sample_idx], 
                        chunk_info['model'].predict,
                        num_features=10
                    )
                    
                    # Display explanation
                    st.write("Top 10 Features for this prediction:")
                    
                    # Create a DataFrame for better display
                    lime_results = []
                    for feature, weight in exp.as_list():
                        lime_results.append({
                            'Feature': feature,
                            'Weight': weight,
                            'Abs_Weight': abs(weight)
                        })
                    
                    lime_df = pd.DataFrame(lime_results)
                    lime_df = lime_df.sort_values('Abs_Weight', ascending=False)
                    
                    st.dataframe(lime_df[['Feature', 'Weight']])
                    
                    # Show actual vs predicted
                    actual = y_actual.iloc[sample_idx]
                    predicted = chunk_info['model'].predict([X_scaled[sample_idx]])[0]
                    st.write(f"**Actual:** {actual:.4f}, **Predicted:** {predicted:.4f}")
                    
                    # Create visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    features = [item[0] for item in exp.as_list()]
                    weights = [item[1] for item in exp.as_list()]
                    
                    colors = ['red' if w < 0 else 'green' for w in weights]
                    ax.barh(range(len(features)), weights, color=colors)
                    ax.set_yticks(range(len(features)))
                    ax.set_yticklabels(features)
                    ax.set_xlabel('LIME Weight')
                    ax.set_title(f'Sample {sample_idx+1} - LIME Explanation')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    st.write("---")
                
            except Exception as e:
                st.error(f"Error in LIME analysis for chunk {chunk_idx+1}: {str(e)}")
                continue
    
    def _prepare_data_for_analysis(self, df, start_row=2, tick_interval=100):
        """
        Prepare data for analysis starting from row 2 with 100-tick intervals.
        """
        print(f"Preparing data for LIME analysis...")
        print(f"Data shape: {df.shape}")
        print(f"Starting from row: {start_row}")
        print(f"Tick interval: {tick_interval}")
        
        # Start from row 2 (index 1)
        analysis_data = []
        
        for i in range(start_row, len(df), tick_interval):
            end_row = min(i + tick_interval, len(df))
            
            # Get data chunk
            chunk = df.iloc[i:end_row].copy()
            
            # Get tick number (assuming it's the row index)
            tick_number = i
            
            # Load appropriate model
            model_data = self.load_model_for_tick(tick_number)
            
            if model_data is not None:
                model, scaler = self.extract_model_and_scaler(model_data)
                
                if model is not None:
                    analysis_data.append({
                        'start_row': i,
                        'end_row': end_row,
                        'tick_number': tick_number,
                        'model': model,
                        'scaler': scaler,
                        'data_chunk': chunk
                    })
        
        print(f"Prepared {len(analysis_data)} analysis chunks")
        return analysis_data

def main():
    """Main function for LIME analysis with saved models."""
    st.title("LIME Analysis with Saved Models")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="lime_uploader")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Shape: {df.shape}")
            
            # Initialize analyzer
            analyzer = LIMEAnalyzer()
            
            # Display model information
            st.subheader("Model Information")
            st.write(f"Found {len(analyzer.model_files)} model files")
            st.write("Model range:", analyzer.model_files[0].split('_')[-1].split('.')[0], 
                    "to", analyzer.model_files[-1].split('_')[-1].split('.')[0])
            
            # Parameters
            start_row = st.number_input("Start Row", min_value=2, value=2, step=1)
            tick_interval = st.number_input("Tick Interval", min_value=50, value=100, step=50)
            num_samples = st.number_input("Number of Samples per Chunk", min_value=1, value=3, step=1)
            
            if st.button("Run LIME Analysis"):
                analyzer.perform_lime_analysis(df, start_row, tick_interval, num_samples)
        
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

if __name__ == "__main__":
    main() 