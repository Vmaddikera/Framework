import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# VIF Analysis imports
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# LIME Analysis imports (needed for model training)
from sklearn.linear_model import SGDRegressor

# LIME Analysis imports
import lime
import lime.lime_tabular
from lime import lime_tabular



EXCLUDE_FEATURES = ['actual_target', 'forecast', 'signal_correct', 'return_pct']

def clean_for_vif(df):
    initial_shape = df.shape
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='any')
    final_shape = df.shape
    if initial_shape != final_shape:
        st.warning(f"Dropped rows/columns with NaN or inf values for VIF calculation. Data shape changed from {initial_shape} to {final_shape}.")
    return df

class VIFLimeAnalyzer:
    """Comprehensive VIF and LIME analysis for trading strategy features."""
    
    def __init__(self, df):
        self.df = df
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.model = None
        self.explainer = None
        self.analysis_successful = False
        
    def validate_data(self):
        if self.df is None:
            st.error("No data provided for analysis!")
            return False
        if len(self.df) == 0:
            st.error("Empty dataset provided!")
            return False
        required_cols = ['y_actual', 'pnl', 'tp_sl_hit_status']
        available_cols = [col for col in required_cols if col in self.df.columns]
        if len(available_cols) == 0:
            st.error("No target columns found! Need at least one of: y_actual, pnl, tp_sl_hit_status")
            return False
        st.success(f"Data validation passed. Found {len(self.df)} samples with {len(self.df.columns)} columns")
        return True
        
    def prepare_data(self):
        try:
            if not self.validate_data():
                return False
                
            exclude_cols = ['y_actual', 'y_predict', 'pnl', 'tp_sl_hit_status', 'Category', 'datetime'] + EXCLUDE_FEATURES
            self.feature_names = [col for col in self.df.columns if col not in exclude_cols]
            
            if len(self.feature_names) == 0:
                st.error("No feature columns found after excluding target columns!")
                return False
                
            st.info(f"Found {len(self.feature_names)} potential feature columns")
            X = self.df[self.feature_names].copy()
            
            numeric_columns = []
            for col in X.columns:
                try:
                    if X[col].dtype == 'object':
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    if pd.api.types.is_numeric_dtype(X[col]):
                        numeric_columns.append(col)
                    else:
                        st.warning(f"Dropping non-numeric column: {col}")
                except Exception as e:
                    st.warning(f"Dropping problematic column {col}: {str(e)}")
            
            X = X[numeric_columns]
            if len(X.columns) == 0:
                st.error("No numeric feature columns found!")
                return False
                
            X = X.fillna(X.mean())
            
            constant_cols = [col for col in X.columns if X[col].std() == 0]
            if constant_cols:
                st.warning(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
                X = X.drop(columns=constant_cols)
            
            if len(X.columns) == 0:
                st.error("No valid features remaining after preprocessing!")
                return False
            
            y = None
            target_source = None
            
            # Try y_actual first
            if 'y_actual' in self.df.columns:
                try:
                    y_actual_raw = self.df['y_actual'].copy()
                    st.info(f"y_actual unique values in prepare_data: {y_actual_raw.unique()}")
                    
                    # Handle different y_actual formats
                    if y_actual_raw.dtype == 'object' or y_actual_raw.dtype == 'string':
                        y_actual_str = y_actual_raw.astype(str).str.lower()
                        # Try multiple mapping strategies
                        if 'buy' in y_actual_str.values or 'sell' in y_actual_str.values:
                            y = y_actual_str.map({'buy': 1, 'sell': 0, '1': 1, '-1': 0, '0': 0})
                        else:
                            # Try numeric conversion
                            y = pd.to_numeric(y_actual_raw, errors='coerce')
                    else:
                        y = y_actual_raw.copy()
                    
                    # Map -1 to 0 if present
                    y = y.replace({-1: 0})
                    valid_mask = y.isin([0, 1])
                    
                    if valid_mask.sum() > 0:
                        X = X[valid_mask]
                        y = y[valid_mask]
                        target_source = 'y_actual'
                        st.success(f"Using y_actual as target. Valid samples: {len(X)}")
                    else:
                        st.warning(f"No valid y_actual values found after mapping. Original values: {y_actual_raw.unique()}")
                        y = None
                except Exception as e:
                    st.warning(f"Error processing y_actual: {str(e)}")
                    y = None
            
            # Try pnl if y_actual failed
            if y is None and 'pnl' in self.df.columns:
                try:
                    y = self.df['pnl'].copy()
                    if y.dtype == object:
                        y = y.map({'+ve': 1, '-ve': -1, '1': 1, '-1': -1}).fillna(0).astype(int)
                    target_source = 'pnl'
                    st.success(f"Using pnl as target. Samples: {len(X)}")
                except Exception as e:
                    st.warning(f"Error processing pnl: {str(e)}")
                    y = None
            
            # Try tp_sl_hit_status if others failed
            if y is None and 'tp_sl_hit_status' in self.df.columns:
                try:
                    tp_sl = self.df['tp_sl_hit_status'].fillna('No TP/SL')
                    y = (tp_sl.str.lower() != 'no tp/sl').astype(int)
                    target_source = 'tp_sl_hit_status'
                    st.success(f"Using tp_sl_hit_status as target. Samples: {len(X)}")
                except Exception as e:
                    st.warning(f"Error processing tp_sl_hit_status: {str(e)}")
                    y = None
            
            if y is None:
                st.error("No valid target variable could be prepared!")
                return False
            
            if len(X) == 0 or len(y) == 0:
                st.error("No valid samples remaining after target preparation!")
                return False
                
            if len(X) != len(y):
                st.error(f"Feature and target length mismatch: X={len(X)}, y={len(y)}")
                return False
            
            self.X = X
            self.y = y
            
            st.success(f" Data preparation successful! {len(X)} samples with {len(X.columns)} features. Target: {target_source}")
            return True
            
        except Exception as e:
            st.error(f"Error in data preparation: {str(e)}")
            return False
    
    def vif_analysis(self):
        st.header(" VIF (Variance Inflation Factor) Analysis")
        try:
            if self.X is None or len(self.X) == 0:
                st.error("No valid features available for VIF analysis!")
                return None
            
            self.X = clean_for_vif(self.X)
            if len(self.X) < len(self.X.columns) + 1:
                st.warning(f"Not enough samples for VIF analysis. Need at least {len(self.X.columns) + 1} samples, have {len(self.X)}")
                return None
            
            vif_data = []
            X_temp = self.X.copy()
            for i in range(len(X_temp.columns)):
                try:
                    vif = variance_inflation_factor(X_temp.values, i)
                    vif_data.append({'Feature': X_temp.columns[i], 'VIF': vif})
                except Exception as e:
                    st.warning(f"Could not calculate VIF for {X_temp.columns[i]}: {str(e)}")
                    continue
            
            if len(vif_data) == 0:
                st.error("No VIF values could be calculated!")
                return None
            
            vif_df = pd.DataFrame(vif_data)
            vif_df = vif_df.sort_values('VIF', ascending=False)
            
            st.subheader("VIF Results")
            st.dataframe(vif_df)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            top_15_vif = vif_df.head(min(15, len(vif_df)))
            colors = ['red' if vif > 10 else 'orange' if vif > 5 else 'green' for vif in top_15_vif['VIF']]
            
            ax1.barh(range(len(top_15_vif)), top_15_vif['VIF'], color=colors)
            ax1.set_yticks(range(len(top_15_vif)))
            ax1.set_yticklabels(top_15_vif['Feature'])
            ax1.set_xlabel('VIF Value')
            ax1.set_title('Top Features by VIF')
            ax1.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='VIF > 5 (Moderate)')
            ax1.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='VIF > 10 (High)')
            ax1.legend()
            
            ax2.hist(vif_df['VIF'], bins=min(20, len(vif_df)), alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='VIF > 5')
            ax2.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='VIF > 10')
            ax2.set_xlabel('VIF Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of VIF Values')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            high_vif = vif_df[vif_df['VIF'] > 10]
            moderate_vif = vif_df[(vif_df['VIF'] > 5) & (vif_df['VIF'] <= 10)]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Features", len(vif_df))
            with col2:
                st.metric("High VIF (>10)", len(high_vif))
            with col3:
                st.metric("Moderate VIF (5-10)", len(moderate_vif))
            
            if len(high_vif) > 0:
                st.warning(f" {len(high_vif)} features have high VIF (>10), indicating multicollinearity:")
                st.write(high_vif[['Feature', 'VIF']].to_string(index=False))
            
            return vif_df
            
        except Exception as e:
            st.error(f"Error in VIF analysis: {str(e)}")
            return None
    


    def lime_analysis(self):
        """Perform LIME (Local Interpretable Model-agnostic Explanations) analysis with error handling."""
        st.header(" LIME (Local Interpretable Model-agnostic Explanations) Analysis")
        try:
            if self.X is None or self.y is None:
                st.error("No valid data available for LIME analysis!")
                return
            
            X_clean = self.X.copy()
            y_clean = self.y.copy()
            
            X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
            X_clean = X_clean.fillna(X_clean.mean())
            
            for col in X_clean.columns:
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
            X_clean = X_clean.fillna(X_clean.mean())
            
            X_scaled = self.scaler.fit_transform(X_clean.values)
            
            # Use only SGDRegressor for all cases
            self.model = SGDRegressor(loss='squared_error', random_state=42, max_iter=1000)
            model_type = "Regression"
            
            self.model.fit(X_scaled, y_clean)
            st.success(f"SGDRegressor trained successfully for LIME analysis")
            st.info(f"Using SGDRegressor for regression with probability wrapper")
            
            X_clean = self.X.copy()
            X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
            X_clean = X_clean.fillna(X_clean.mean())
            
            for col in X_clean.columns:
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
            X_clean = X_clean.fillna(X_clean.mean())
            
            X_scaled = self.scaler.transform(X_clean.values)
            
            feature_names = [str(name) for name in self.X.columns.tolist()]
            
            # Create probability wrapper for SGDRegressor
            def predict_proba_wrapper(X):
                predictions = self.model.predict(X)
                # Convert regression predictions to probability-like scores
                # For binary targets, map to [0,1] range
                if len(np.unique(self.y)) == 2:
                    # Normalize predictions to [0,1] range
                    min_pred = predictions.min()
                    max_pred = predictions.max()
                    if max_pred != min_pred:
                        normalized = (predictions - min_pred) / (max_pred - min_pred)
                    else:
                        normalized = np.full_like(predictions, 0.5)
                    # Return as 2D array for LIME compatibility
                    return np.column_stack([1 - normalized, normalized])
                else:
                    # For regression, return predictions as single column
                    return predictions.reshape(-1, 1)
            
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_scaled,
                feature_names=feature_names,
                mode='regression'
            )
            
            st.subheader("LIME Analysis for Time Series Samples")
            st.info(" **Time Series Analysis**: Using sequential sampling (beginning, middle, end) instead of random sampling to preserve temporal order.")
            
            if len(self.X) == 0:
                st.error("No data available for LIME analysis!")
                return
            
            num_samples = min(3, len(self.X))
            
            if len(self.X) >= 3:
                sample_indices = [0, len(self.X) // 2, len(self.X) - 1]
            else:
                sample_indices = list(range(len(self.X)))
            
            for i, idx in enumerate(sample_indices):
                st.write(f"**Sample {i+1} (Time Series Position: {idx + 1}/{len(self.X)})**")
                if i == 0:
                    st.write(" **Time Series Position**: Beginning")
                elif i == len(sample_indices) - 1:
                    st.write(" **Time Series Position**: End")
                else:
                    st.write(" **Time Series Position**: Middle")
                
                try:
                    exp = explainer.explain_instance(
                        X_scaled[idx], 
                        predict_proba_wrapper,
                        num_features=min(10, len(self.X.columns))
                    )
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    top_features = exp.as_list()[:5]
                    feature_names = [f[0] for f in top_features]
                    feature_values = [f[1] for f in top_features]
                    colors = ['red' if v < 0 else 'green' for v in feature_values]
                    
                    display_names = []
                    for name in feature_names:
                        if len(name) > 20:
                            display_names.append(name[:17] + "...")
                        else:
                            display_names.append(name)
                    
                    ax1.barh(range(len(feature_names)), [abs(v) for v in feature_values], color=colors)
                    ax1.set_yticks(range(len(feature_names)))
                    ax1.set_yticklabels(display_names)
                    ax1.set_xlabel('LIME Weight')
                    ax1.set_title(f"LIME Explanation - Sample {i+1}")
                    
                    sample_features = self.X.iloc[idx]
                    
                    ax2.scatter(range(len(feature_names)), feature_values, color=colors, s=100)
                    ax2.set_xticks(range(len(feature_names)))
                    ax2.set_xticklabels(display_names, rotation=45, ha='right')
                    ax2.set_ylabel('LIME Weight')
                    ax2.set_title(f'Feature Impact Values - Sample {i+1}')
                    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.write("**Actual Feature Values:**")
                    st.write(f"**Available features in data:** {list(sample_features.index)}")
                    
                    for feature, value in top_features:
                        base_feature = feature.split(' <= ')[0].split(' >= ')[0].split(' == ')[0].split(' != ')[0]
                        
                        found_feature = None
                        actual_value = None
                        
                        if base_feature in sample_features.index:
                            found_feature = base_feature
                            actual_value = sample_features[base_feature]
                        else:
                            try:
                                if base_feature.startswith('feature_'):
                                    feature_idx = int(base_feature.split('_')[1])
                                    if feature_idx < len(sample_features.index):
                                        found_feature = sample_features.index[feature_idx]
                                        actual_value = sample_features.iloc[feature_idx]
                            except:
                                pass
                            
                            if found_feature is None:
                                for col in sample_features.index:
                                    if base_feature.lower() in col.lower() or col.lower() in base_feature.lower():
                                        found_feature = col
                                        actual_value = sample_features[col]
                                        break
                        
                        try:
                            if found_feature is not None and actual_value is not None:
                                st.write(f"- {feature}: {actual_value:.4f} (LIME weight: {value:.4f})")
                            else:
                                st.write(f"- {feature}: Feature not found in original data (LIME weight: {value:.4f})")
                        except Exception as e:
                            st.write(f"- {feature}: Error accessing feature (LIME weight: {value:.4f})")
                    
                    st.divider()
                except Exception as e:
                    st.error(f"Error processing LIME analysis for sample {i+1}: {str(e)}")
                    continue
            
            st.subheader("LIME Analysis Summary")
            st.write("""
            **LIME Analysis Insights (SGD Model):**
            - Shows how each feature contributes to the SGD model's prediction for specific instances
            - Positive weights (green) indicate features that push the prediction toward the positive class
            - Negative weights (red) indicate features that push the prediction toward the negative class
            - The magnitude shows the strength of each feature's influence
            - Using SGD for faster training and linear interpretability
            """)
        except Exception as e:
            st.error(f"Error in LIME analysis: {str(e)}")
    
    def run_complete_analysis(self):
        st.title(" Advanced Feature Analysis: VIF & LIME")
        try:
            if not self.prepare_data():
                st.error(" Data preparation failed. Analysis cannot proceed.")
                return
            
            if self.X is None or len(self.X) == 0:
                st.error(" No valid features found for analysis!")
                return
                
            if self.y is None or len(self.y) == 0:
                st.error(" No valid target variable found for analysis!")
                return
            
            tab1, tab2 = st.tabs(["VIF Analysis", "LIME Analysis"])
            
            vif_results = None
            
            with tab1:
                vif_results = self.vif_analysis()
            
            with tab2:
                self.lime_analysis()
            
            self.analysis_successful = True
            
        except Exception as e:
            st.error(f" Error in complete analysis: {str(e)}")
            st.info("This analysis requires specific data format. Please ensure your CSV has numeric features and target columns (y_actual, pnl, etc.).")

def main(df=None):
    if df is None:
        st.error("Please upload a CSV file first!")
        return
    
    try:
        if not isinstance(df, pd.DataFrame):
            st.error("Invalid data format. Expected pandas DataFrame.")
            return
        
        st.info(f"Data shape: {df.shape}")
        st.info(f"Available columns: {list(df.columns)}")
        
        analyzer = VIFLimeAnalyzer(df)
        analyzer.run_complete_analysis()
    except Exception as e:
        st.error(f"Error in VIF & LIME Analysis: {str(e)}")
        st.info("This analysis requires specific data format. Please ensure your CSV has numeric features and target columns (y_actual, pnl, etc.).")
        return

if __name__ == "__main__":
    st.title("VIF & LIME Analysis")
    st.write("This module provides comprehensive feature analysis using VIF and LIME techniques with SGD models.") 
