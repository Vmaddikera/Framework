import pandas as pd
import numpy as np
import pickle
import os
import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
from scipy.special import softmax
import matplotlib.pyplot as plt
import streamlit as st
import lime
import lime.lime_tabular
from pathlib import Path

# Set matplotlib backend for Streamlit
import matplotlib
matplotlib.use('Agg')

class SavedModelAnalyzer:
    """
    Analyzer for SHAP and LIME analysis using saved ML models from pickle files.
    Models are loaded every 100 ticks and used for analysis.
    """
    
    def __init__(self, models_dir="model_evaluation/models"):
        self.models_dir = models_dir
        self.model_files = self._get_model_files()
        self.models_info = self._analyze_models()
        
    def _get_model_files(self):
        """Get all model pickle files sorted by tick number."""
        pattern = os.path.join(self.models_dir, "model_tick_*.pkl")
        model_files = glob.glob(pattern)
        
        # Sort by tick number
        model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        print(f"Found {len(model_files)} model files")
        return model_files
    
    def _analyze_models(self):
        """Analyze the model files to understand the structure."""
        models_info = []
        
        for model_file in self.model_files[:5]:  # Analyze first 5 models
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                tick_num = int(model_file.split('_')[-1].split('.')[0])
                
                # Analyze model structure
                if isinstance(model_data, dict):
                    model_info = {
                        'tick': tick_num,
                        'file': model_file,
                        'type': 'dict',
                        'keys': list(model_data.keys()) if isinstance(model_data, dict) else None
                    }
                else:
                    model_info = {
                        'tick': tick_num,
                        'file': model_file,
                        'type': type(model_data).__name__,
                        'keys': None
                    }
                
                models_info.append(model_info)
                
            except Exception as e:
                print(f"Error analyzing {model_file}: {e}")
        
        return models_info
    
    def load_model_for_tick(self, tick_number):
        """
        Load the appropriate model for a given tick number.
        Models are saved every 100 ticks, so we find the closest model.
        """
        # Find the model file for this tick range
        # Models are saved every 100 ticks, so tick 119500-119599 uses model_tick_119500.pkl
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
    
    def prepare_data_for_analysis(self, df, start_row=2, tick_interval=100):
        """
        Prepare data for analysis starting from row 2 with 100-tick intervals.
        """
        print(f"Preparing data for analysis...")
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
    
    def perform_shap_analysis(self, df, start_row=2, tick_interval=100):
        """
        Perform SHAP analysis using saved models for every 100 ticks.
        """
        st.header("SHAP Analysis with Saved Models")
        
        # Prepare data for analysis
        analysis_chunks = self.prepare_data_for_analysis(df, start_row, tick_interval)
        
        if not analysis_chunks:
            st.error("No analysis chunks prepared. Check model files and data.")
            return
        
        # Collect SHAP results
        all_shap_values = []
        all_feature_names = []
        all_predictions = []
        all_actuals = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, chunk_info in enumerate(analysis_chunks):
            status_text.text(f"Processing chunk {idx+1}/{len(analysis_chunks)} (ticks {chunk_info['tick_number']}-{chunk_info['tick_number']+tick_interval})")
            
            try:
                # Get feature columns (exclude target and prediction columns)
                feature_cols = [col for col in chunk_info['data_chunk'].columns 
                              if col not in ['y_actual', 'y_predict', 'forecast', 'actual_target', 'signal_correct', 'executed_signal']]
                
                X = chunk_info['data_chunk'][feature_cols]
                y_actual = chunk_info['data_chunk']['y_actual']
                
                # Apply scaler if available
                if chunk_info['scaler'] is not None:
                    X_scaled = chunk_info['scaler'].transform(X)
                else:
                    X_scaled = X
                
                # Get predictions from the model
                model = chunk_info['model']
                y_pred = model.predict(X_scaled)
                
                # Create SHAP explainer
                explainer = shap.Explainer(model.predict, X_scaled)
                shap_values = explainer(X_scaled)
                
                # Store results
                all_shap_values.append(shap_values.values)
                all_feature_names.append(feature_cols)
                all_predictions.append(y_pred)
                all_actuals.append(y_actual.values)
                
                # Calculate metrics for this chunk
                mae = mean_absolute_error(y_actual, y_pred)
                mse = mean_squared_error(y_actual, y_pred)
                r2 = r2_score(y_actual, y_pred)
                
                st.write(f"Chunk {idx+1} Metrics:")
                st.write(f"- MAE: {mae:.4f}")
                st.write(f"- MSE: {mse:.4f}")
                st.write(f"- R²: {r2:.4f}")
                
            except Exception as e:
                st.error(f"Error processing chunk {idx+1}: {str(e)}")
                continue
            
            progress_bar.progress((idx + 1) / len(analysis_chunks))
        
        # Aggregate results
        if all_shap_values:
            st.success("SHAP Analysis completed!")
            
            # Combine all SHAP values
            combined_shap_values = np.vstack(all_shap_values)
            combined_feature_names = all_feature_names[0]  # Use first chunk's feature names
            
            # Calculate feature importance
            feature_importance = np.mean(np.abs(combined_shap_values), axis=0)
            feature_importance_dict = dict(zip(combined_feature_names, feature_importance))
            
            # Sort by importance
            sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Display results
            st.subheader("Feature Importance (SHAP)")
            
            # Create feature importance plot
            fig, ax = plt.subplots(figsize=(12, 8))
            features, importances = zip(*sorted_features[:20])  # Top 20 features
            ax.barh(range(len(features)), importances)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('SHAP Importance')
            ax.set_title('Feature Importance from Saved Models')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Display feature importance table
            st.subheader("Top 20 Most Important Features")
            importance_df = pd.DataFrame(sorted_features[:20], columns=['Feature', 'SHAP Importance'])
            st.dataframe(importance_df)
            
            # Overall metrics
            combined_predictions = np.concatenate(all_predictions)
            combined_actuals = np.concatenate(all_actuals)
            
            overall_mae = mean_absolute_error(combined_actuals, combined_predictions)
            overall_mse = mean_squared_error(combined_actuals, combined_predictions)
            overall_r2 = r2_score(combined_actuals, combined_predictions)
            
            st.subheader("Overall Performance Metrics")
            st.write(f"- Overall MAE: {overall_mae:.4f}")
            st.write(f"- Overall MSE: {overall_mse:.4f}")
            st.write(f"- Overall R²: {overall_r2:.4f}")
            
            return feature_importance_dict
        
        else:
            st.error("No SHAP values calculated. Check model compatibility.")
            return None
    
    def perform_lime_analysis(self, df, start_row=2, tick_interval=100, num_samples=5):
        """
        Perform LIME analysis using saved models for every 100 ticks.
        """
        st.header("LIME Analysis with Saved Models")
        
        # Prepare data for analysis
        analysis_chunks = self.prepare_data_for_analysis(df, start_row, tick_interval)
        
        if not analysis_chunks:
            st.error("No analysis chunks prepared. Check model files and data.")
            return
        
        # Select a few chunks for LIME analysis (to avoid overwhelming output)
        selected_chunks = analysis_chunks[:3]  # First 3 chunks
        
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
                
                # Analyze a few samples
                for sample_idx in range(min(num_samples, len(X_scaled))):
                    st.write(f"Sample {sample_idx+1} Analysis:")
                    
                    # Get LIME explanation
                    exp = explainer.explain_instance(
                        X_scaled[sample_idx], 
                        chunk_info['model'].predict,
                        num_features=10
                    )
                    
                    # Display explanation
                    st.write("**Top 10 Features for this prediction:**")
                    for feature, weight in exp.as_list():
                        st.write(f"- {feature}: {weight:.4f}")
                    
                    # Show actual vs predicted
                    actual = y_actual.iloc[sample_idx]
                    predicted = chunk_info['model'].predict([X_scaled[sample_idx]])[0]
                    st.write(f"Actual: {actual:.4f}, Predicted: {predicted:.4f}")
                    st.write("---")
                
            except Exception as e:
                st.error(f"Error in LIME analysis for chunk {chunk_idx+1}: {str(e)}")
                continue

def main():
    """Main function for SHAP and LIME analysis with saved models."""
    st.title("SHAP & LIME Analysis with Saved Models")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="saved_models_uploader")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Shape: {df.shape}")
            
            # Initialize analyzer
            analyzer = SavedModelAnalyzer()
            
            # Display model information
            st.subheader("Model Information")
            st.write(f"Found {len(analyzer.model_files)} model files")
            st.write("Model range:", analyzer.model_files[0].split('_')[-1].split('.')[0], 
                    "to", analyzer.model_files[-1].split('_')[-1].split('.')[0])
            
            # Analysis options
            analysis_type = st.selectbox(
                "Choose Analysis Type",
                ["SHAP Analysis", "LIME Analysis", "Both"]
            )
            
            # Parameters
            start_row = st.number_input("Start Row", min_value=2, value=2, step=1)
            tick_interval = st.number_input("Tick Interval", min_value=50, value=100, step=50)
            
            if st.button("Run Analysis"):
                if analysis_type in ["SHAP Analysis", "Both"]:
                    analyzer.perform_shap_analysis(df, start_row, tick_interval)
                
                if analysis_type in ["LIME Analysis", "Both"]:
                    analyzer.perform_lime_analysis(df, start_row, tick_interval)
        
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

if __name__ == "__main__":
    main() 