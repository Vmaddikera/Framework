import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
from scipy.special import softmax
import matplotlib.pyplot as plt
import streamlit as st
import feature_weighting

# Set matplotlib backend for Streamlit
import matplotlib
matplotlib.use('Agg')

def print_feature_importances_shap_values(shap_values, features):
    importances = [np.mean(np.abs(shap_values.values[:, i])) for i in range(shap_values.values.shape[1])]
    importances_norm = softmax(importances)
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
    feature_importances_norm = dict(sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse=True))
    print("SHAP Feature Importances:")
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print('Regression evaluation metrics:')
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

def main():
    # === INPUT your CSV file path here ===
    csv_file = r"C:\Users\lenovo\Downloads\Framework\target_corr_feat_USDCAD_model_output.csv"
    
    # Load data, skipping the first row to start from second row (index 1)
    df = pd.read_csv(csv_file, skiprows=[0])
    df.columns = df.columns.str.strip()

    # Check that target column exists
    target_col = 'y_actual'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")

    # Feature columns: all columns up to (but excluding) y_actual
    feature_cols = list(df.columns[:df.columns.get_loc(target_col)])

    # Extract features and target
    X = df[feature_cols]
    y = df[target_col]

    print(f"Features used ({len(feature_cols)}): {feature_cols}")
    print(f"Target variable: {target_col}")
    print(f"Dataset shape: {df.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit SGD regressor
    model = SGDRegressor(random_state=42, max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)
    evaluate_regression(y_test, y_pred)

    # SHAP explainer setup
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_test)

    # Print feature importances based on SHAP values
    print_feature_importances_shap_values(shap_values, X.columns)

    # Plot SHAP summary bar and beeswarm plots
    shap.plots.bar(shap_values, max_display=20)
    plt.show()

    shap.plots.beeswarm(shap_values, max_display=20)
    plt.show()

def shap_analysis_y_actual_streamlit(df):
    """Perform SHAP analysis specifically for y_actual target using SGDRegressor with Streamlit."""
    st.header("SHAP Analysis for y_actual Target")
    try:
        if 'y_actual' not in df.columns:
            st.error("y_actual column not found in the dataset!")
            return
        
        # st.info("Found y_actual column. Preparing data for SHAP analysis...")
        
        # First, let's examine the y_actual values to understand the format
        y_actual_raw = df['y_actual'].copy()
        # st.info(f"y_actual unique values: {y_actual_raw.unique()}")
        # st.info(f"y_actual data type: {y_actual_raw.dtype}")
        
        # Define excluded columns (same as in VIF/LIME analysis)
        EXCLUDE_FEATURES = ['actual_target', 'forecast', 'signal_correct', 'return_pct']
        exclude_cols = ['y_actual', 'y_predict', 'pnl', 'tp_sl_hit_status', 'tp_sl_status', 'duration', 'timestamp', 'Category', 'datetime'] + EXCLUDE_FEATURES
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            st.error("No feature columns found after excluding target columns!")
            return
        
        # st.info(f"Feature columns (excluded y_actual): {feature_cols}")
        # st.info(f"Total columns in dataset: {list(df.columns)}")
        
        # Check which columns have NaN values in the original dataset
        nan_columns = df.columns[df.isnull().any()].tolist()
        st.info(f"Columns with NaN values in original dataset: {nan_columns}")
        
        # Show sample of problematic columns
        if nan_columns:
            st.info("Sample of problematic columns:")
            for col in nan_columns[:5]:  # Show first 5
                if col in df.columns:
                    st.write(f"- {col}: {df[col].isnull().sum()} NaN values out of {len(df)}")
        
        X = df[feature_cols].copy()
        y_actual = df['y_actual'].copy()
        
        # st.info(f"Original y_actual shape: {y_actual.shape}")
        # st.info(f"Original X shape: {X.shape}")
        
        # Convert features to numeric and handle NaN values more carefully
        numeric_columns = []
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    if pd.api.types.is_numeric_dtype(X[col]):
                        numeric_columns.append(col)
                    else:
                        st.warning(f"Dropping non-numeric column {col}")
                except:
                    st.warning(f"Dropping problematic column {col}")
            else:
                numeric_columns.append(col)
        
        X = X[numeric_columns]
        # st.info(f"After keeping only numeric columns: {X.shape}")
        # st.info(f"Numeric columns: {numeric_columns}")
        
        # Handle inf and NaN values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with mean for each column
        for col in X.columns:
            if X[col].isnull().any():
                col_mean = X[col].mean()
                if pd.isna(col_mean):
                    # If mean is NaN, use 0
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(col_mean)
        
        # st.info(f"After cleaning X shape: {X.shape}")
        # st.info(f"X columns with NaN: {X.columns[X.isnull().any()].tolist()}")
        
        # Check if there are still any NaN values
        if X.isnull().any().any():
            st.warning("Still have NaN values after cleaning, dropping those rows")
            X = X.dropna()
            # st.info(f"After dropping NaN rows, X shape: {X.shape}")
        
        # Ensure y_actual and X have the same index
        common_index = X.index.intersection(y_actual.index)
        st.info(f"Common index length: {len(common_index)}")
        
        if len(common_index) == 0:
            st.error("No common indices between X and y_actual after cleaning!")
            st.info(f"X index range: {X.index.min()} to {X.index.max()}")
            st.info(f"y_actual index range: {y_actual.index.min()} to {y_actual.index.max()}")
            return
        
        X = X.loc[common_index]
        y_actual = y_actual.loc[common_index]
        
        st.info(f"After alignment, y_actual shape: {y_actual.shape}")
        st.info(f"y_actual values after alignment: {y_actual.unique()}")
        st.info(f"X shape after alignment: {X.shape}")
        
        # Handle different y_actual formats
        y_actual_mapped = None
        
        st.info(f"y_actual dtype: {y_actual.dtype}")
        st.info(f"y_actual unique values before mapping: {y_actual.unique()}")
        
        # Try different mapping strategies
        if y_actual.dtype == 'object' or y_actual.dtype == 'string':
            # Handle string values like 'buy', 'sell', '1', '-1', etc.
            y_actual_str = y_actual.astype(str).str.lower()
            if 'buy' in y_actual_str.values or 'sell' in y_actual_str.values:
                y_actual_mapped = y_actual_str.map({'buy': 1, 'sell': 0, '1': 1, '-1': 0, '0': 0})
            else:
                # Try numeric conversion
                y_actual_mapped = pd.to_numeric(y_actual, errors='coerce')
        else:
            # Handle numeric values
            y_actual_mapped = y_actual.copy()
        
        st.info(f"y_actual_mapped unique values after initial mapping: {y_actual_mapped.unique()}")
        
        # Ensure we have binary values (0 and 1)
        if y_actual_mapped is not None:
            # Map -1 to 0 if present
            y_actual_mapped = y_actual_mapped.replace({-1: 0})
            st.info(f"y_actual_mapped after replacing -1 with 0: {y_actual_mapped.unique()}")
            
            # Keep only 0 and 1 values
            valid_mask = y_actual_mapped.isin([0, 1])
            st.info(f"Valid mask sum: {valid_mask.sum()}")
            st.info(f"Valid mask shape: {valid_mask.shape}")
            
            if valid_mask.sum() == 0:
                st.error(f"No valid y_actual values found after mapping. Original values: {y_actual.unique()}")
                st.error("Expected values: -1/1, 0/1, or 'buy'/'sell'")
                return
            
            X = X[valid_mask]
            y_actual_mapped = y_actual_mapped[valid_mask]
            st.info(f"Final X shape after filtering: {X.shape}")
            st.info(f"Final y_actual_mapped shape: {y_actual_mapped.shape}")
        else:
            st.error("Failed to map y_actual values to binary format")
            return
        
        st.success(f"Prepared {len(X)} samples with {len(X.columns)} features for y_actual SHAP analysis")
        st.info(f"y_actual_mapped unique values: {y_actual_mapped.unique()}")
        st.info(f"y_actual_mapped value counts: {y_actual_mapped.value_counts()}")
        
        # Show which features are being used for SHAP analysis
        st.info(f"Features used for SHAP analysis: {list(X.columns)}")
        st.info(f"Excluded features: {exclude_cols}")
        
        # Train/test split for SHAP analysis
        X_train, X_test, y_train, y_test = train_test_split(X, y_actual_mapped, test_size=0.2, random_state=42)
        
        # Train SGDRegressor
        model = SGDRegressor(random_state=42, max_iter=1000, tol=1e-3)
        model.fit(X_train, y_train)
        st.success("SGDRegressor trained successfully for y_actual SHAP analysis")
        
        # Evaluate model performance
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # st.subheader("Model Performance Metrics")
        # col1, col2, col3, col4 = st.columns(4)
        # with col1:
        #     st.metric("MAE", f"{mae:.4f}")
        # with col2:
        #     st.metric("MSE", f"{mse:.4f}")
        # with col3:
        #     st.metric("RMSE", f"{rmse:.4f}")
        # with col4:
        #     st.metric("R²", f"{r2:.4f}")
        
        # SHAP explainer setup
        import shap
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(X_test)
        
        # Calculate feature importances based on SHAP values
        importances = [np.mean(np.abs(shap_values.values[:, i])) for i in range(shap_values.values.shape[1])]
        
        # Show raw importance values for debugging
        st.info(f"Raw SHAP importance values (top 10): {importances[:10]}")
        st.info(f"Total features analyzed: {len(importances)}")
        st.info(f"Max importance: {max(importances):.6f}, Min importance: {min(importances):.6f}")
        st.info(f"Importance range: {max(importances) - min(importances):.6f}")
        
        importances_norm = softmax(importances)
        feature_importances = {fea: imp for imp, fea in zip(importances, X.columns)}
        feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, X.columns)}
        feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
        feature_importances_norm = dict(sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse=True))
        
        st.subheader("SHAP Feature Importances")
        st.write("**Top 20 Features by SHAP Importance:**")
        
        # Display feature importances
        importance_data = []
        for i, (feature, importance) in enumerate(list(feature_importances.items())[:20]):
            importance_data.append({
                'Rank': i + 1,
                'Feature': feature,
                'SHAP_Importance': importance,
                'Softmax_Importance': feature_importances_norm[feature]
            })
        
        importance_df = pd.DataFrame(importance_data)
        st.dataframe(importance_df, use_container_width=True)
        
        # Create SHAP plots
        st.subheader("SHAP Visualization")
        
        # SHAP Summary Bar Plot
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.plots.bar(shap_values, max_display=20, show=False)
            plt.title("SHAP Feature Importance (Bar Plot)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.success("SHAP Bar Plot created successfully")
        except Exception as e:
            st.error(f"Error creating SHAP Bar Plot: {str(e)}")
        
        # SHAP Beeswarm Plot
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.plots.beeswarm(shap_values, max_display=20, show=False)
            plt.title("SHAP Feature Impact (Beeswarm Plot)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.success("SHAP Beeswarm Plot created successfully")
        except Exception as e:
            st.error(f"Error creating SHAP Beeswarm Plot: {str(e)}")
        
        # SHAP Waterfall Plot for a sample
        # if len(X_test) > 0:
        #     try:
        #         st.subheader("SHAP Waterfall Plot (Sample Prediction)")
        #         sample_idx = 0  # First sample
        #         fig, ax = plt.subplots(figsize=(12, 8))
        #         shap.plots.waterfall(shap_values[sample_idx], max_display=15, show=False)
        #         plt.title(f"SHAP Waterfall Plot - Sample {sample_idx + 1}")
        #         plt.tight_layout()
        #         st.pyplot(fig)
        #         plt.close()
                
        #         st.write(f"**Sample {sample_idx + 1} Details:**")
        #         st.write(f"- Actual y_actual value: {y_test.iloc[sample_idx]}")
        #         st.write(f"- Predicted value: {y_pred[sample_idx]:.4f}")
        #         st.write(f"- SHAP base value: {shap_values.base_values[sample_idx]:.4f}")
        #         st.success("SHAP Waterfall Plot created successfully")
        #     except Exception as e:
        #         st.error(f"Error creating SHAP Waterfall Plot: {str(e)}")
        

        
        # Feature Weighting System for SHAP
        st.subheader(" SHAP-Based Feature Weighting")
        weighting_system = feature_weighting.FeatureWeightingSystem()
        
        # Create SHAP importance dictionary
        shap_importance_dict = {feature: importance for feature, importance in feature_importances.items()}
        
        # Create tabs for different weighting approaches
        weight_tab1, weight_tab2 = st.tabs([
            "Standard SHAP Weighting",
            "Optimal Trading Weights"
        ])
        
        with weight_tab1:
            st.write("**Standard SHAP-Based Feature Weighting:**")
            # Display standard weighting interface
            shap_weights = weighting_system.display_weighting_interface(
                shap_importance_dict=shap_importance_dict,
                analysis_type="shap_analysis"
            )
            
            # Show weighted feature summary
            if shap_weights:
                summary = weighting_system.get_weighted_feature_summary(shap_weights)
                st.write("**Weighted Feature Summary:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Features", summary['total_features'])
                with col2:
                    st.metric("Total Weight", f"{summary['total_weight']:.4f}")
                with col3:
                    st.metric("Max Weight", f"{summary['max_weight']:.4f}")
                with col4:
                    st.metric("Mean Weight", f"{summary['mean_weight']:.4f}")
                
                st.write("**Top 5 Weighted Features:**")
                top_features_df = pd.DataFrame(summary['top_5_features'], columns=['Feature', 'Weight'])
                st.dataframe(top_features_df, use_container_width=True)
        
        with weight_tab2:
            st.write("**Optimal Trading Weights (SHAP-Optimized):**")
            
            # Market volatility selection
            market_volatility = st.selectbox(
                "Select Market Volatility Level:",
                ['low', 'medium', 'high'],
                index=1,
                help="Low: Stable markets, Medium: Normal volatility, High: Volatile markets (earnings, Fed meetings, etc.)",
                key="shap_optimal_trading_volatility"
            )
            
            # Calculate optimal trading weights using SHAP data
            optimal_shap_weights = weighting_system.calculate_optimal_trading_weights(
                shap_importance_dict=shap_importance_dict,
                market_volatility=market_volatility
            )
            
            if optimal_shap_weights:
                # Display optimal weights
                st.write("**Optimal Trading Feature Weights (SHAP-Based):**")
                optimal_df = pd.DataFrame(list(optimal_shap_weights.items()), columns=['Feature', 'Weight'])
                optimal_df = optimal_df.sort_values('Weight', ascending=False)
                
                # Show top features
                st.write("**Top 20 Features by Optimal Trading Weight:**")
                st.dataframe(optimal_df.head(20), use_container_width=True)
                
                # Weight distribution plot
                fig, ax = plt.subplots(figsize=(12, 6))
                top_20_weights = optimal_df.head(20)
                colors = plt.cm.viridis(np.linspace(0, 1, len(top_20_weights)))
                bars = ax.bar(range(len(top_20_weights)), top_20_weights['Weight'], color=colors)
                ax.set_xlabel('Features')
                ax.set_ylabel('Weight')
                ax.set_title(f'Optimal SHAP Trading Weight Distribution (Top 20) - {market_volatility.title()} Volatility')
                ax.set_xticks(range(len(top_20_weights)))
                ax.set_xticklabels(top_20_weights['Feature'], rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Weight statistics
                st.write("**Optimal Weight Statistics:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Weight", f"{sum(optimal_shap_weights.values()):.4f}")
                with col2:
                    st.metric("Max Weight", f"{max(optimal_shap_weights.values()):.4f}")
                with col3:
                    st.metric("Min Weight", f"{min(optimal_shap_weights.values()):.4f}")
                with col4:
                    st.metric("Weight Std", f"{np.std(list(optimal_shap_weights.values())):.4f}")
                
                # Trading recommendations
                st.write("**Trading Strategy Recommendations:**")
                top_5_features = optimal_df.head(5)
                st.write("**Top 5 Features for Trading Strategy:**")
                for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
                    st.write(f"{i}. **{row['Feature']}** (Weight: {row['Weight']:.4f})")
                
                # Market volatility insights
                volatility_insights = {
                    'low': "Stable market conditions - focus on consistent, reliable features",
                    'medium': "Normal market volatility - balanced approach with moderate concentration",
                    'high': "High volatility - aggressive concentration on top features, reduce noise"
                }
                st.info(f"**Market Volatility Insight:** {volatility_insights[market_volatility]}")
                
                # Model performance comparison
                st.write("**SHAP vs Optimal Trading Weights Comparison:**")
                comparison_data = []
                for feature in optimal_df.head(10)['Feature']:
                    original_weight = shap_importance_dict.get(feature, 0)
                    optimal_weight = optimal_shap_weights.get(feature, 0)
                    comparison_data.append({
                        'Feature': feature,
                        'Original SHAP': original_weight,
                        'Optimal Trading': optimal_weight,
                        'Improvement': optimal_weight - original_weight
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Save optimal weights
                if st.button(" Save Optimal SHAP Trading Weights", key="save_optimal_shap_trading"):
                    weighting_system.save_weights(optimal_shap_weights, f"optimal_shap_trading_{market_volatility}")
                
                # Export optimal weights
                if st.button(" Export Optimal SHAP Weights as JSON", key="export_optimal_shap_trading"):
                    import json
                    weights_json = json.dumps(optimal_shap_weights, indent=2)
                    st.download_button(
                        label="Download Optimal SHAP Trading Weights JSON",
                        data=weights_json,
                        file_name=f"optimal_shap_trading_weights_{market_volatility}.json",
                        mime="application/json",
                        key="download_optimal_shap_trading"
                    )
        
        st.subheader("SHAP Analysis Summary")
        st.write("""
        **SHAP Analysis Insights for y_actual:**
        - **Global Feature Importance**: Shows overall feature importance across all samples
        - **Direction**: Red (negative) and blue (positive) show feature impact direction
        - **Magnitude**: Bar height shows feature importance strength
        - **Individual Impact**: Each point shows how a feature affects a specific prediction
        - **Model Interpretability**: Explains how SGDRegressor makes decisions for y_actual predictions
        - **Feature Weighting**: Use the weights above to improve your model without removing features
        """)
        
    except Exception as e:
        st.error(f"Error in y_actual SHAP analysis: {str(e)}")
        st.info("Make sure you have the 'shap' package installed: pip install shap")

if __name__ == "__main__":
    main()
