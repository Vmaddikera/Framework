import streamlit as st
import pandas as pd
import sys
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the path to import strategy_analysis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy_analysis import analyze_trades

def main(df):
    """Strategy Analysis Streamlit Module"""
    st.header("Strategy Analysis")
    
    if df is None:
        st.warning("Please upload a CSV file first.")
        return
    
    # Check if required columns exist
    required_columns = ['y_actual', 'y_predict', 'pnl', 'tp_sl_status']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.info("Available columns: " + ", ".join(df.columns))
        return
    
    # Create a temporary file for the analysis
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        df.to_csv(tmp_file.name, index=False)
        tmp_path = tmp_file.name
    
    try:
        # Run the analysis
        result_df, counts, confusion_matrices = analyze_trades(tmp_path)
        
        if result_df is not None:
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Category Distribution")
                st.dataframe(counts)
                
                # Show key metrics
                st.subheader("Key Metrics")
                y_actual = result_df['y_actual']
                y_pred = result_df['y_predict']
                ml_accuracy = (y_actual == y_pred).mean()
                positive_pnl_rate = (result_df['pnl'] > 0).mean()
                tp_sl_hit_rate = (result_df['tp_sl_status'].str.lower() != 'no tp/sl').mean()
                
                st.metric("ML Model Accuracy", f"{ml_accuracy:.2%}")
                st.metric("Positive PnL Rate", f"{positive_pnl_rate:.2%}")
                st.metric("TP/SL Hit Rate", f"{tp_sl_hit_rate:.2%}")
            
            with col2:
                st.subheader("Value Distribution")
                st.write("**y_actual value counts:**")
                st.write(result_df['y_actual'].value_counts().sort_index())
                st.write("**y_predict value counts:**")
                st.write(result_df['y_predict'].value_counts().sort_index())
            
            # Show confusion matrices if available
            if confusion_matrices:
                st.subheader("Confusion Matrices")
                labels = confusion_matrices.get('labels', [-1, 0, 1])
                label_names = []
                for label in labels:
                    if label == 1:
                        label_names.append('Buy (1)')
                    elif label == -1:
                        label_names.append('Sell (-1)')
                    elif label == 0:
                        label_names.append('Neutral (0)')
                    else:
                        label_names.append(f'Class {label}')

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # ML Model Accuracy
                sns.heatmap(confusion_matrices['ml_accuracy'], annot=True, fmt='d', cmap='Blues',
                            xticklabels=label_names, yticklabels=label_names, ax=axes[0])
                axes[0].set_title('ML Model Accuracy\n(Actual vs Predicted)')
                axes[0].set_xlabel('Predicted')
                axes[0].set_ylabel('Actual')

                # ML Success vs PnL
                sns.heatmap(confusion_matrices['ml_success_vs_pnl'], annot=True, fmt='d', cmap='Greens',
                            xticklabels=['Negative PnL', 'Positive PnL'],
                            yticklabels=['ML Failed', 'ML Success'], ax=axes[1])
                axes[1].set_title('ML Success vs PnL')
                axes[1].set_xlabel('PnL')
                axes[1].set_ylabel('ML Success')

                # TP/SL vs PnL
                sns.heatmap(confusion_matrices['tp_sl_vs_pnl'], annot=True, fmt='d', cmap='Reds',
                            xticklabels=['Negative PnL', 'Positive PnL'],
                            yticklabels=['No TP/SL', 'TP/SL Hit'], ax=axes[2])
                axes[2].set_title('TP/SL Hit vs PnL')
                axes[2].set_xlabel('PnL')
                axes[2].set_ylabel('TP/SL Status')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Download results
            st.subheader("Download Results")
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download Analysis Results",
                data=csv,
                file_name="strategy_analysis_results.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error in Strategy Analysis: {str(e)}")
        st.info("Please check your data format and ensure all required columns are present.")
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass 