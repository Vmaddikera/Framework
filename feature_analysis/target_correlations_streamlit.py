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
import os
import feature_weighting

EXCLUDE_FEATURES = ['actual_target', 'forecast', 'signal_correct', 'return_pct']

def get_feature_columns(df, target_columns):
    return [col for col in df.columns if col not in EXCLUDE_FEATURES and col not in target_columns]

class TargetCorrelationAnalyzerDirect:
    """Direct DataFrame-based target correlation analysis for Streamlit."""
    
    def __init__(self, df):
        self.df = df
        self.X = None
        self.feature_names = None
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset."""
        st.write("Preparing data...")
        
        # Identify feature columns (exclude target, metadata, and explicitly excluded columns)
        exclude_cols = ['y_actual', 'y_predict', 'pnl', 'tp_sl_status', 'Category', 'datetime'] + EXCLUDE_FEATURES
        self.feature_names = [col for col in self.df.columns if col not in exclude_cols]
        
        if len(self.feature_names) == 0:
            st.error("No feature columns found!")
            return None
        
        st.write(f"Found {len(self.feature_names)} feature columns")
        
        # Prepare features
        X = self.df[self.feature_names].copy()
        
        # Convert to numeric and handle missing values
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    st.warning(f"Dropping non-numeric column {col}")
                    X = X.drop(columns=[col])
        
        X = X.fillna(X.mean())
        
        self.X = X
        
        st.success(f"Prepared {len(X)} samples with {len(X.columns)} features")
        return X
    
    def get_target_data(self, target_name):
        """Get specific target data for correlation analysis."""
        if target_name == 'y_actual':
            # Use 1/-1 directly if present, else map buy/sell
            target = self.df['y_actual']
            if target.dtype == object:
                # Try to map buy/sell to 1/-1
                target = target.astype(str).str.lower().map({'buy': 1, 'sell': -1, '1': 1, '-1': -1})
            else:
                target = pd.to_numeric(target, errors='coerce')
        elif target_name == 'pnl':
            target = self.df['pnl'].copy()
            if target.dtype == object:
                target = target.map({'+ve': 1, '-ve': -1, '1': 1, '-1': -1}).fillna(0).astype(int)
        elif target_name == 'tp_sl':
            target = self.df['tp_sl_status'].fillna('No TP/SL')
            target = (target.str.lower() != 'no tp/sl').astype(int)
        else:
            raise ValueError(f"Unknown target: {target_name}")
        return target
    
    def analyze_target_correlations(self, X, target, target_label, plot_color, dist_color):
        st.subheader(f"{target_label.upper()} CORRELATION ANALYSIS")
        # Remove rows with invalid target values
        valid_mask = ~target.isnull()
        X_valid = X[valid_mask]
        target_valid = target[valid_mask]

        if X_valid.empty or X_valid.isnull().values.all():
            st.error(f"No valid features available for {target_label} correlation analysis after filtering. Please check your data.")
            return

        correlations = []
        for feature in X_valid.columns:
            if X_valid[feature].isnull().all():
                st.warning(f"Feature {feature} is all NaN and will be skipped.")
                continue
            corr = X_valid[feature].corr(target_valid)
            correlations.append({
                'Feature': feature,
                'Correlation': corr,
                'Abs_Correlation': abs(corr)
            })

        correlation_df = pd.DataFrame(correlations)
        correlation_df = correlation_df.sort_values('Abs_Correlation', ascending=False)

        st.write(f"Top 15 features by absolute correlation with {target_label}:")
        st.dataframe(correlation_df.head(15))

        if not correlation_df.empty and correlation_df['Abs_Correlation'].notnull().any():
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            top_15_features = correlation_df.head(15)
            colors = ['red' if x < 0 else plot_color for x in top_15_features['Correlation']]
            ax1.barh(range(len(top_15_features)), top_15_features['Abs_Correlation'], color=colors)
            ax1.set_yticks(range(len(top_15_features)))
            ax1.set_yticklabels(top_15_features['Feature'])
            ax1.set_xlabel('Absolute Correlation')
            ax1.set_title(f'Top 15 Features by Correlation with {target_label}')
            ax2.hist(correlation_df['Correlation'].dropna(), bins=20, alpha=0.7, color=dist_color)
            ax2.set_xlabel(f'Correlation with {target_label}')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Distribution of Feature Correlations with {target_label}')
            top_10_features = correlation_df.head(10)['Feature'].tolist()
            if len(top_10_features) > 1:
                correlation_matrix = X_valid[top_10_features].corr()
                if not correlation_matrix.isnull().values.all():
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
                    ax3.set_title(f'Correlation Matrix of Top 10 Features')
                else:
                    ax3.set_title('Correlation Matrix: All NaN')
            top_10_features = correlation_df.head(10)
            ax4.bar(range(len(top_10_features)), top_10_features['Abs_Correlation'])
            ax4.set_xticks(range(len(top_10_features)))
            ax4.set_xticklabels(top_10_features['Feature'], rotation=45, ha='right')
            ax4.set_ylabel('Absolute Correlation')
            ax4.set_title(f'Top 10 Features by Correlation Strength')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.error(f"No valid correlations to plot for {target_label}.")

        return {
            'correlation_df': correlation_df,
            'top_feature': correlation_df.iloc[0]['Feature'] if len(correlation_df) > 0 else None
        }

    def analyze_y_actual_correlations(self, X):
        """Analyze correlations between features and y_actual target."""
        y_actual = self.df['y_actual']
        if y_actual.dtype == object:
            y_actual = y_actual.astype(str).str.lower().map({'buy': 1, 'sell': -1, '1': 1, '-1': -1})
        else:
            y_actual = pd.to_numeric(y_actual, errors='coerce')
        valid_mask = y_actual.isin([-1, 1])
        return self.analyze_target_correlations(X[valid_mask], y_actual[valid_mask], "y_actual", "blue", "skyblue")

    def analyze_pnl_correlations(self, X):
        """Analyze correlations between features and PnL target."""
        pnl = self.df['pnl'].copy()
        if pnl.dtype == object:
            pnl = pnl.map({'+ve': 1, '-ve': -1, '1': 1, '-1': -1}).fillna(0).astype(int)
        return self.analyze_target_correlations(X, pnl, "PnL", "blue", "lightgreen")

    def analyze_tp_sl_correlations(self, X):
        """Analyze correlations between features and TP/SL target."""
        tp_sl = self.df['tp_sl_status'].fillna('No TP/SL')
        tp_sl_binary = (tp_sl.str.lower() != 'no tp/sl').astype(int)
        return self.analyze_target_correlations(X, tp_sl_binary, "TP/SL Hit", "blue", "purple")
    
    def compare_target_correlations(self, y_actual_corr, pnl_corr, tp_sl_corr):
        """Compare feature correlations across all targets (no duplicate plotting)."""
        st.subheader("CROSS-TARGET CORRELATION COMPARISON")
        # Merge correlation DataFrames
        merged = pd.DataFrame()
        if y_actual_corr and 'correlation_df' in y_actual_corr:
            merged['Feature'] = y_actual_corr['correlation_df']['Feature']
            merged['y_actual_corr'] = y_actual_corr['correlation_df']['Correlation']
        if pnl_corr and 'correlation_df' in pnl_corr:
            if merged.empty:
                merged['Feature'] = pnl_corr['correlation_df']['Feature']
            merged['pnl_corr'] = pnl_corr['correlation_df']['Correlation']
        if tp_sl_corr and 'correlation_df' in tp_sl_corr:
            if merged.empty:
                merged['Feature'] = tp_sl_corr['correlation_df']['Feature']
            merged['tp_sl_corr'] = tp_sl_corr['correlation_df']['Correlation']
        # Calculate average correlation
        if not merged.empty:
            merged['avg_corr'] = merged[['y_actual_corr', 'pnl_corr', 'tp_sl_corr']].mean(axis=1)
            merged = merged.sort_values('avg_corr', ascending=False)
            st.write("Top 15 features by average correlation across all targets:")
            st.dataframe(merged.head(15))
        else:
            st.warning("No valid features for cross-target correlation comparison.")
        return {
            'comparison_df': merged
        }
    
    def run_complete_analysis(self):
        """Run complete target correlation analysis (no duplicate plots/results)."""
        st.write("Starting comprehensive target correlation analysis...")
        # Load and prepare data
        X = self.load_and_prepare_data()
        if X is None:
            return None
        # Check for empty or all-NaN features
        if X.empty or X.isnull().values.all():
            st.error("No valid features available for correlation analysis after filtering. Please check your data.")
            return
        # Run all correlation analyses ONCE
        y_actual_results = self.analyze_y_actual_correlations(X)
        pnl_results = self.analyze_pnl_correlations(X)
        tp_sl_results = self.analyze_tp_sl_correlations(X)
        # Pass results to comparison (no duplicate analysis)
        comparison_results = self.compare_target_correlations(
            y_actual_results, pnl_results, tp_sl_results
        )
        # Combine results
        results = {
            'y_actual': y_actual_results,
            'pnl': pnl_results,
            'tp_sl': tp_sl_results,
            'comparison': comparison_results,
            'data_info': {
                'total_samples': len(X),
                'total_features': len(X.columns)
            }
        }
        st.success("Target correlation analysis completed!")
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", results['data_info']['total_samples'])
        with col2:
            st.metric("y_actual Top Feature", results['y_actual']['top_feature'])
        with col3:
            st.metric("PnL Top Feature", results['pnl']['top_feature'])
        # Feature Weighting System
        st.subheader(" Feature Weighting System")
        weighting_system = feature_weighting.FeatureWeightingSystem()
        
        # Get correlation data for weighting
        y_actual_corr_df = results['y_actual']['correlation_df']
        pnl_corr_df = results['pnl']['correlation_df']
        
        # Create tabs for different weighting approaches
        weight_tab1, weight_tab2, weight_tab3, weight_tab4 = st.tabs([
            "y_actual Weighting",
            "PnL Weighting", 
            "Combined Weighting",
            "Optimal Trading Weights"
        ])
        
        with weight_tab1:
            st.write("**Optimal Feature Weighting based on y_actual Correlations:**")
            y_actual_weights = weighting_system.display_weighting_interface(
                correlation_df=y_actual_corr_df,
                analysis_type="y_actual_correlation"
            )
        
        with weight_tab2:
            st.write("**Optimal Feature Weighting based on PnL Correlations:**")
            pnl_weights = weighting_system.display_weighting_interface(
                correlation_df=pnl_corr_df,
                analysis_type="pnl_correlation"
            )
        
        with weight_tab3:
            st.write("**Optimal Combined Feature Weighting (y_actual + PnL):**")
            # Combine correlation data
            combined_corr_df = pd.concat([
                y_actual_corr_df.assign(Target='y_actual'),
                pnl_corr_df.assign(Target='pnl')
            ]).groupby('Feature').agg({
                'Abs_Correlation': 'mean',
                'Correlation': 'mean'
            }).reset_index()
            combined_corr_df = combined_corr_df.sort_values('Abs_Correlation', ascending=False)
            
            combined_weights = weighting_system.display_weighting_interface(
                correlation_df=combined_corr_df,
                analysis_type="combined_correlation"
            )
        
        with weight_tab4:
            st.write("**Optimal Trading Weights (Market-Optimized):**")
            
            # Market volatility selection
            market_volatility = st.selectbox(
                "Select Market Volatility Level:",
                ['low', 'medium', 'high'],
                index=1,
                help="Low: Stable markets, Medium: Normal volatility, High: Volatile markets (earnings, Fed meetings, etc.)",
                key="target_corr_optimal_trading_volatility"
            )
            
            # Calculate optimal trading weights
            optimal_weights = weighting_system.calculate_optimal_trading_weights(
                correlation_df=combined_corr_df,
                market_volatility=market_volatility
            )
            
            if optimal_weights:
                # Display optimal weights
                st.write("**Optimal Trading Feature Weights:**")
                optimal_df = pd.DataFrame(list(optimal_weights.items()), columns=['Feature', 'Weight'])
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
                ax.set_title(f'Optimal Trading Weight Distribution (Top 20) - {market_volatility.title()} Volatility')
                ax.set_xticks(range(len(top_20_weights)))
                ax.set_xticklabels(top_20_weights['Feature'], rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Weight statistics
                st.write("**Optimal Weight Statistics:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Weight", f"{sum(optimal_weights.values()):.4f}")
                with col2:
                    st.metric("Max Weight", f"{max(optimal_weights.values()):.4f}")
                with col3:
                    st.metric("Min Weight", f"{min(optimal_weights.values()):.4f}")
                with col4:
                    st.metric("Weight Std", f"{np.std(list(optimal_weights.values())):.4f}")
                
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
                
                # Save optimal weights
                if st.button(" Save Optimal Trading Weights", key="save_optimal_trading"):
                    weighting_system.save_weights(optimal_weights, f"optimal_trading_{market_volatility}")
                
                # Export optimal weights
                if st.button(" Export Optimal Weights as JSON", key="export_optimal_trading"):
                    import json
                    weights_json = json.dumps(optimal_weights, indent=2)
                    st.download_button(
                        label="Download Optimal Trading Weights JSON",
                        data=weights_json,
                        file_name=f"optimal_trading_weights_{market_volatility}.json",
                        mime="application/json",
                        key="download_optimal_trading"
                    )
        
        # Display detailed results
        st.subheader("Detailed Results")
        # y_actual Results
        with st.expander("y_actual Correlation Results"):
            st.write("**Top 10 Features by Correlation with y_actual:**")
            y_actual_df = results['y_actual']['correlation_df'].head(10)
            st.dataframe(y_actual_df, use_container_width=True)
        # PnL Results
        with st.expander("PnL Correlation Results"):
            st.write("**Top 10 Features by Correlation with PnL:**")
            pnl_df = results['pnl']['correlation_df'].head(10)
            st.dataframe(pnl_df, use_container_width=True)
        # TP/SL Results
        with st.expander("TP/SL Correlation Results"):
            st.write("**Top 10 Features by Correlation with TP/SL:**")
            tp_sl_df = results['tp_sl']['correlation_df'].head(10)
            st.dataframe(tp_sl_df, use_container_width=True)
        # Comparison Results
        with st.expander("Cross-Target Comparison Results"):
            st.write("**Top 10 Features by Average Correlation:**")
            comparison_df = results['comparison']['comparison_df'].head(10)
            st.dataframe(comparison_df, use_container_width=True)
        return results

def main(df=None):
    """Target Correlations Analysis for Streamlit."""
    
    st.header("Target Correlations")
    
    if df is None:
        st.warning("Please upload a CSV file first.")
        return
    
    try:
        # Validate that df is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            st.error("Invalid data format. Expected pandas DataFrame.")
            return
        
        # Show data info
        st.info(f"Data shape: {df.shape}")
        st.info(f"Available columns: {list(df.columns)}")
        
        # Check for required columns with better error handling
        required_columns = ['y_actual', 'pnl']
        available_columns = [col for col in required_columns if col in df.columns]
        
        if len(available_columns) == 0:
            st.error("No required target columns found!")
            st.info(f"Required columns: {required_columns}")
            st.info(f"Available columns: {list(df.columns)}")
            return
        
        st.success(f"Found target columns: {available_columns}")
        
        # Run analysis automatically (no button needed)
        with st.spinner("Running target correlation analysis..."):
            try:
                # Create analyzer and run analysis
                analyzer = TargetCorrelationAnalyzerDirect(df)
                results = analyzer.run_complete_analysis()
                
                if results is None:
                    st.error("Analysis failed. Please check your data format.")
                    return
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error("Please check your data format and try again.")
    
    except Exception as e:
        st.error(f"Error in Target Correlations Analysis: {str(e)}")
        st.info("Please check your data format and try again.")

if __name__ == "__main__":
    st.title("Target Correlations Analysis")
    st.write("This module provides comprehensive target correlation analysis for trading strategy features.") 