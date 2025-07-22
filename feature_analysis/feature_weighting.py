import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
import json
import os

class FeatureWeightingSystem:
    """Feature weighting system for trading strategy analysis."""
    
    def __init__(self):
        self.weights_file = "feature_weights.json"
        self.saved_weights = self.load_saved_weights()
    
    def load_saved_weights(self):
        """Load previously saved feature weights."""
        try:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            st.warning(f"Could not load saved weights: {str(e)}")
        return {}
    
    def save_weights(self, weights_dict, analysis_type):
        """Save feature weights to file."""
        try:
            self.saved_weights[analysis_type] = weights_dict
            with open(self.weights_file, 'w') as f:
                json.dump(self.saved_weights, f, indent=2)
            st.success(f"Feature weights saved for {analysis_type} analysis!")
        except Exception as e:
            st.error(f"Error saving weights: {str(e)}")
    
    def calculate_correlation_weights(self, correlation_df, method='softmax', custom_weights=None):
        """
        Calculate feature weights based on correlation analysis.
        
        Args:
            correlation_df (pd.DataFrame): DataFrame with correlation results
            method (str): Weighting method ('softmax', 'linear', 'custom')
            custom_weights (dict): Custom weights if method is 'custom'
        
        Returns:
            dict: Feature weights
        """
        if correlation_df.empty:
            return {}
        
        # Use absolute correlation values for weighting
        abs_correlations = correlation_df['Abs_Correlation'].values
        features = correlation_df['Feature'].values
        
        # Convert to numpy array for calculations
        abs_correlations = np.array(abs_correlations)
        
        if method == 'softmax':
            # Use very low temperature for aggressive weight concentration
            temperature = 0.1  # Much lower temperature = very concentrated weights
            weights = softmax(abs_correlations / temperature)
        elif method == 'linear':
            # Use squared correlations for more aggressive weighting
            weights = (abs_correlations ** 2) / (abs_correlations ** 2).sum()
        elif method == 'custom' and custom_weights:
            # Use custom weights provided by user
            weights = []
            for feature in features:
                weights.append(custom_weights.get(feature, 0.01))  # Default small weight
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
        else:
            # Default to aggressive softmax
            temperature = 0.1
            weights = softmax(abs_correlations / temperature)
        
        return dict(zip(features, weights))
    
    def calculate_shap_weights(self, shap_importance_dict, method='softmax', custom_weights=None):
        """
        Calculate feature weights based on SHAP importance.
        
        Args:
            shap_importance_dict (dict): Dictionary of feature:importance pairs
            method (str): Weighting method ('softmax', 'linear', 'custom')
            custom_weights (dict): Custom weights if method is 'custom'
        
        Returns:
            dict: Feature weights
        """
        if not shap_importance_dict:
            return {}
        
        features = list(shap_importance_dict.keys())
        importances = list(shap_importance_dict.values())
        
        # Convert to numpy array for calculations
        importances = np.array(importances)
        
        if method == 'softmax':
            # Use very low temperature for aggressive weight concentration
            temperature = 0.1  # Much lower temperature = very concentrated weights
            weights = softmax(importances / temperature)
        elif method == 'linear':
            # Use squared importance for more aggressive weighting
            weights = (importances ** 2) / (importances ** 2).sum()
        elif method == 'custom' and custom_weights:
            weights = []
            for feature in features:
                weights.append(custom_weights.get(feature, 0.01))
            weights = np.array(weights)
            weights = weights / weights.sum()
        else:
            # Default to aggressive softmax
            temperature = 0.1
            weights = softmax(importances / temperature)
        
        return dict(zip(features, weights))
    
    def combine_weights(self, correlation_weights, shap_weights, correlation_weight=0.5):
        """
        Combine weights from correlation and SHAP analysis.
        
        Args:
            correlation_weights (dict): Weights from correlation analysis
            shap_weights (dict): Weights from SHAP analysis
            correlation_weight (float): Weight given to correlation analysis (0-1)
        
        Returns:
            dict: Combined feature weights
        """
        combined_weights = {}
        all_features = set(correlation_weights.keys()) | set(shap_weights.keys())
        
        for feature in all_features:
            corr_weight = correlation_weights.get(feature, 0)
            shap_weight = shap_weights.get(feature, 0)
            
            # Weighted combination
            combined_weight = (correlation_weight * corr_weight + 
                             (1 - correlation_weight) * shap_weight)
            combined_weights[feature] = combined_weight
        
        # Normalize to sum to 1
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            combined_weights = {k: v/total_weight for k, v in combined_weights.items()}
        
        return combined_weights
    
    def display_weighting_interface(self, correlation_df=None, shap_importance_dict=None, 
                                  analysis_type="combined"):
        """
        Display automatic weighting interface in Streamlit.
        
        Args:
            correlation_df (pd.DataFrame): Correlation analysis results
            shap_importance_dict (dict): SHAP importance results
            analysis_type (str): Type of analysis for saving weights
        """
        st.subheader(" Automatic Feature Weighting System")
        st.write("""
        **Purpose**: Automatically assign optimal distributed weights to features based on correlation and SHAP analysis.
        This allows you to improve your model without removing features directly.
        """)
        
        # Automatic method selection based on data availability
        if correlation_df is not None and shap_importance_dict is not None:
            st.write("**Combined Analysis (Correlation + SHAP):**")
            method = "combined_optimal"
            
            # Calculate individual weights using optimal methods
            corr_weights = self.calculate_correlation_weights(correlation_df, 'softmax')
            shap_weights = self.calculate_shap_weights(shap_importance_dict, 'softmax')
            
            # Optimal combination (60% SHAP, 40% correlation for better model performance)
            correlation_weight = 0.4
            weights = self.combine_weights(corr_weights, shap_weights, correlation_weight)
            
            # Display comparison
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Correlation Weights:**")
                corr_df = pd.DataFrame(list(corr_weights.items()), columns=['Feature', 'Weight'])
                st.dataframe(corr_df.head(10), use_container_width=True)
            
            with col2:
                st.write("**SHAP Weights:**")
                shap_df = pd.DataFrame(list(shap_weights.items()), columns=['Feature', 'Weight'])
                st.dataframe(shap_df.head(10), use_container_width=True)
        
        elif correlation_df is not None:
            st.write("**Correlation-Based Weighting:**")
            method = "correlation_optimal"
            weights = self.calculate_correlation_weights(correlation_df, 'softmax')
        
        elif shap_importance_dict is not None:
            st.write("**SHAP-Based Weighting:**")
            method = "shap_optimal"
            weights = self.calculate_shap_weights(shap_importance_dict, 'softmax')
        
        else:
            st.warning("No analysis data available for weighting.")
            return
        
        if weights:
            # Display weights
            st.write("**Optimal Feature Weights (Automatically Calculated):**")
            weights_df = pd.DataFrame(list(weights.items()), columns=['Feature', 'Weight'])
            weights_df = weights_df.sort_values('Weight', ascending=False)
            
            # Show top features
            st.write("**Top 20 Features by Optimal Weight:**")
            st.dataframe(weights_df.head(20), use_container_width=True)
            
            # Weight distribution plot
            fig, ax = plt.subplots(figsize=(12, 6))
            top_20_weights = weights_df.head(20)
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_20_weights)))
            bars = ax.bar(range(len(top_20_weights)), top_20_weights['Weight'], color=colors)
            ax.set_xlabel('Features')
            ax.set_ylabel('Weight')
            ax.set_title('Optimal Feature Weight Distribution (Top 20)')
            ax.set_xticks(range(len(top_20_weights)))
            ax.set_xticklabels(top_20_weights['Feature'], rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Weight statistics
            st.write("**Weight Statistics:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Weight", f"{sum(weights.values()):.4f}")
            with col2:
                st.metric("Max Weight", f"{max(weights.values()):.4f}")
            with col3:
                st.metric("Min Weight", f"{min(weights.values()):.4f}")
            with col4:
                st.metric("Weight Std", f"{np.std(list(weights.values())):.4f}")
            
            # Model improvement recommendations
            st.write("**Model Improvement Recommendations:**")
            top_5_features = weights_df.head(5)
            st.write("**Top 5 Features to Focus On:**")
            for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
                st.write(f"{i}. **{row['Feature']}** (Weight: {row['Weight']:.4f})")
            
            # Save weights automatically
            if st.button(" Save Optimal Feature Weights", key=f"save_{analysis_type}"):
                self.save_weights(weights, analysis_type)
            
            # Export weights
            if st.button(" Export Weights as JSON", key=f"export_{analysis_type}"):
                weights_json = json.dumps(weights, indent=2)
                st.download_button(
                    label="Download Optimal Weights JSON",
                    data=weights_json,
                    file_name=f"optimal_feature_weights_{analysis_type}.json",
                    mime="application/json",
                    key=f"download_{analysis_type}"
                )
            
            # Show saved weights
            if self.saved_weights:
                st.write("**Previously Saved Weights:**")
                for saved_type, saved_weights in self.saved_weights.items():
                    with st.expander(f"Saved {saved_type} weights"):
                        saved_df = pd.DataFrame(list(saved_weights.items()), columns=['Feature', 'Weight'])
                        st.dataframe(saved_df.head(10), use_container_width=True)
            
            return weights
        
        return weights
    
    def apply_weights_to_features(self, X, weights):
        """
        Apply feature weights to dataset.
        
        Args:
            X (pd.DataFrame): Feature dataset
            weights (dict): Feature weights
        
        Returns:
            pd.DataFrame: Weighted features
        """
        X_weighted = X.copy()
        
        for feature, weight in weights.items():
            if feature in X_weighted.columns:
                X_weighted[feature] = X_weighted[feature] * weight
        
        return X_weighted
    
    def get_weighted_feature_summary(self, weights):
        """
        Get summary of weighted features.
        
        Args:
            weights (dict): Feature weights
        
        Returns:
            dict: Summary statistics
        """
        if not weights:
            return {}
        
        weight_values = list(weights.values())
        
        return {
            'total_features': len(weights),
            'total_weight': sum(weight_values),
            'max_weight': max(weight_values),
            'min_weight': min(weight_values),
            'mean_weight': np.mean(weight_values),
            'std_weight': np.std(weight_values),
            'top_5_features': sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def calculate_optimal_trading_weights(self, correlation_df=None, shap_importance_dict=None, 
                                       market_volatility='medium'):
        """
        Calculate optimal feature weights specifically for stock trading.
        
        Args:
            correlation_df (pd.DataFrame): Correlation analysis results
            shap_importance_dict (dict): SHAP importance results
            market_volatility (str): Market volatility level ('low', 'medium', 'high')
        
        Returns:
            dict: Optimized feature weights for trading
        """
        if not correlation_df and not shap_importance_dict:
            return {}
        
        # Market volatility adjustments
        volatility_settings = {
            'low': {'temperature': 0.2, 'correlation_weight': 0.5, 'power': 1.5},
            'medium': {'temperature': 0.1, 'correlation_weight': 0.4, 'power': 2.0},
            'high': {'temperature': 0.05, 'correlation_weight': 0.3, 'power': 2.5}
        }
        
        settings = volatility_settings.get(market_volatility, volatility_settings['medium'])
        
        weights = {}
        
        if correlation_df is not None and shap_importance_dict is not None:
            # Combined analysis - optimal for trading
            abs_correlations = correlation_df['Abs_Correlation'].values
            features = correlation_df['Feature'].values
            
            # Enhanced correlation weighting with volatility adjustment
            corr_weights = np.power(abs_correlations, settings['power'])
            corr_weights = corr_weights / corr_weights.sum()
            
            # Enhanced SHAP weighting
            shap_features = list(shap_importance_dict.keys())
            shap_importances = np.array(list(shap_importance_dict.values()))
            shap_weights = softmax(shap_importances / settings['temperature'])
            
            # Combine with market-optimized ratios
            combined_weights = {}
            all_features = set(features) | set(shap_features)
            
            for feature in all_features:
                corr_weight = dict(zip(features, corr_weights)).get(feature, 0)
                shap_weight = dict(zip(shap_features, shap_weights)).get(feature, 0)
                
                # Market-optimized combination
                combined_weight = (settings['correlation_weight'] * corr_weight + 
                                 (1 - settings['correlation_weight']) * shap_weight)
                combined_weights[feature] = combined_weight
            
            weights = combined_weights
            
        elif correlation_df is not None:
            # Correlation-only with volatility adjustment
            abs_correlations = correlation_df['Abs_Correlation'].values
            features = correlation_df['Feature'].values
            
            weights_array = np.power(abs_correlations, settings['power'])
            weights_array = softmax(weights_array / settings['temperature'])
            weights = dict(zip(features, weights_array))
            
        elif shap_importance_dict is not None:
            # SHAP-only with volatility adjustment
            features = list(shap_importance_dict.keys())
            importances = np.array(list(shap_importance_dict.values()))
            
            weights_array = softmax(importances / settings['temperature'])
            weights = dict(zip(features, weights_array))
        
        # Final normalization
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights

    def calculate_direct_optimal_trading_weights(self, df, target_columns=['y_actual', 'pnl']):
        """
        Calculate optimal trading weights directly from the dataset without requiring separate correlation/SHAP analysis.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_columns (list): List of target columns to exclude from features
        
        Returns:
            dict: Optimal feature weights for trading
        """
        if df is None or df.empty:
            return {}
        
        # Prepare features
        exclude_features = ['actual_target', 'forecast', 'signal_correct', 'return_pct', 'y_predict'] + target_columns
        feature_columns = [col for col in df.columns if col not in exclude_features]
        
        if not feature_columns:
            return {}
        
        X = df[feature_columns].copy()
        
        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        # Handle NaN values
        X = X.fillna(0)
        
        if X.empty:
            return {}
        
        # Calculate feature importance using multiple methods
        feature_weights = {}
        
        # Method 1: Variance-based importance
        variances = X.var()
        variance_weights = variances / variances.sum()
        
        # Method 2: Correlation with targets (if available)
        correlation_weights = {}
        for target in target_columns:
            if target in df.columns:
                correlations = X.corrwith(df[target]).abs()
                correlation_weights[target] = correlations / correlations.sum()
        
        # Method 3: Statistical significance
        from scipy import stats
        significance_weights = {}
        for col in X.columns:
            try:
                # Calculate statistical significance using t-test
                _, p_value = stats.ttest_1samp(X[col].dropna(), 0)
                significance_weights[col] = 1 - p_value  # Higher significance = higher weight
            except:
                significance_weights[col] = 0.5
        
        # Combine all methods
        for feature in X.columns:
            # Base weight from variance
            base_weight = variance_weights.get(feature, 0)
            
            # Add correlation weight if available
            corr_weight = 0
            for target_weights in correlation_weights.values():
                corr_weight += target_weights.get(feature, 0)
            if correlation_weights:
                corr_weight /= len(correlation_weights)
            
            # Add significance weight
            sig_weight = significance_weights.get(feature, 0.5)
            
            # Combine weights (40% variance, 40% correlation, 20% significance)
            combined_weight = (0.4 * base_weight + 0.4 * corr_weight + 0.2 * sig_weight)
            feature_weights[feature] = combined_weight
        
        # Normalize weights
        total_weight = sum(feature_weights.values())
        if total_weight > 0:
            feature_weights = {k: v/total_weight for k, v in feature_weights.items()}
        
        return feature_weights
    
    def display_simple_optimal_trading_interface(self, df):
        """
        Display simplified optimal trading weights interface.
        
        Args:
            df (pd.DataFrame): Input dataset
        
        Returns:
            dict: Optimal feature weights
        """
        st.subheader("Optimal Trading Weights Dashboard")
        st.write("""
        **Purpose**: This dashboard provides market-optimized feature weights for better trading results.
        Weights are calculated using variance, correlation, and statistical significance analysis.
        """)
        
        if df is None or df.empty:
            st.warning("No data available for analysis.")
            return {}
        
        # Calculate optimal weights directly
        with st.spinner("Calculating optimal trading weights..."):
            optimal_weights = self.calculate_direct_optimal_trading_weights(df)
        
        if not optimal_weights:
            st.error("Could not calculate optimal weights. Please check your data.")
            return {}
        
        # Display results
        st.success("Optimal trading weights calculated successfully!")
        
        # Create weights dataframe
        weights_df = pd.DataFrame(list(optimal_weights.items()), columns=['Feature', 'Weight'])
        weights_df = weights_df.sort_values('Weight', ascending=False)
        
        # Show top features
        st.write("**Top 20 Features by Optimal Trading Weight:**")
        st.dataframe(weights_df.head(20), use_container_width=True)
        
        # Weight distribution plot
        fig, ax = plt.subplots(figsize=(12, 6))
        top_20_weights = weights_df.head(20)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_20_weights)))
        bars = ax.bar(range(len(top_20_weights)), top_20_weights['Weight'], color=colors)
        ax.set_xlabel('Features')
        ax.set_ylabel('Weight')
        ax.set_title('Optimal Trading Weight Distribution (Top 20)')
        ax.set_xticks(range(len(top_20_weights)))
        ax.set_xticklabels(top_20_weights['Feature'], rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Weight statistics
        st.write("**Weight Statistics:**")
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
        top_5_features = weights_df.head(5)
        st.write("**Top 5 Features for Trading Strategy:**")
        for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
            st.write(f"{i}. **{row['Feature']}** (Weight: {row['Weight']:.4f})")
        
        # Save and export options
        col1, col2 = st.columns(2)
        with col1:
            if st.button(" Save Optimal Trading Weights", key="save_simple_optimal"):
                self.save_weights(optimal_weights, "simple_optimal_trading")
                st.success("Weights saved successfully!")
        
        with col2:
            if st.button(" Export as JSON", key="export_simple_optimal"):
                weights_json = json.dumps(optimal_weights, indent=2)
                st.download_button(
                    label="Download Optimal Trading Weights",
                    data=weights_json,
                    file_name="optimal_trading_weights.json",
                    mime="application/json",
                    key="download_simple_optimal"
                )
        
        return optimal_weights

def main():
    """Main function for feature weighting system - called from other analysis modules."""
    st.title("Feature Weighting System")
    st.info("This module is designed to be used with actual analysis results from SHAP and Target Correlation analyses.")
    st.write("Please run the SHAP Analysis or Target Correlations tabs to get feature weighting results.")

if __name__ == "__main__":
    main() 