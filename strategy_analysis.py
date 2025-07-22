import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def categorize_trade(row):
    y_actual = row['y_actual']  # Keep as numeric
    y_pred = row['y_predict']   # Keep as numeric
    tp_sl = str(row['tp_sl_status']).lower()
    pnl = row['pnl']

    # Normalize tp_sl to a standard form
    if pd.isna(tp_sl) or tp_sl in ['none', 'no tp/sl', 'no tp_sl', 'no']:
        tp_sl = 'no tp/sl'

    ml_success = (y_actual == y_pred)
    pnl_positive = (pnl > 0)
    tp_sl_hit = (tp_sl != 'no tp/sl')

    if ml_success:
        if not tp_sl_hit:
            if pnl_positive:
                return "ML Model, Risk Management and Trade Management are good and PnL is positive"
            else:
                return "ML model succeeded, but pnl not positive and no TP/SL hit : Risk Management and Trade Management are Bad"
        else:  # TP or SL hit
            if pnl_positive:
                return "ML Model, Risk Management and Trade Management are good and PnL is positive"
            else:
                return "ML model is good but Risk Management and Trade Management are bad"
    else:
        if pnl_positive:
            return "Unlikely Event(Luck)"
        else:
            return "ML model failed"

def create_confusion_matrices(df):
    """
    Create confusion matrices for different aspects of the trading analysis.
    """
    print("Creating confusion matrices...")
    print(f"Available columns: {list(df.columns)}")
    
    # Check if required columns exist
    if 'y_actual' not in df.columns or 'y_predict' not in df.columns:
        print("Warning: y_actual or y_predict columns not found!")
        print("Available columns:", list(df.columns))
        return None
    
    # Use numeric values directly (1 for buy, -1 for sell)
    y_actual = df['y_actual']
    y_pred = df['y_predict']
    
    print(f"Unique values in y_actual: {y_actual.unique()}")
    print(f"Unique values in y_predict: {y_pred.unique()}")
    
    # Get unique labels from the data
    unique_labels = sorted(list(set(y_actual.unique()) | set(y_pred.unique())))
    print(f"Combined unique labels: {unique_labels}")
    
    # Only create confusion matrix if we have valid labels
    if len(unique_labels) >= 2:
        # Create confusion matrix for ML accuracy
        ml_cm = confusion_matrix(y_actual, y_pred, labels=unique_labels)
        
        # Create confusion matrix for PnL vs ML success
        ml_success = (y_actual == y_pred)
        pnl_positive = (df['pnl'] > 0)
        
        # Create confusion matrix for TP/SL vs PnL
        tp_sl_hit = df['tp_sl_status'].str.lower() != 'no tp/sl'
        
        return {
            'ml_accuracy': ml_cm,
            'ml_success_vs_pnl': confusion_matrix(ml_success, pnl_positive),
            'tp_sl_vs_pnl': confusion_matrix(tp_sl_hit, pnl_positive),
            'labels': unique_labels
        }
    else:
        print("Warning: Not enough unique labels for confusion matrix")
        return None

def plot_confusion_matrices(confusion_matrices, save_path=None):
    """
    Plot all confusion matrices.
    """
    if confusion_matrices is None:
        print("No confusion matrices to plot")
        return
        
    print("Creating confusion matrix plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # ML Accuracy Confusion Matrix
    labels = confusion_matrices.get('labels', [-1, 0, 1])
    
    # Create proper label names based on the actual labels
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
    
    sns.heatmap(confusion_matrices['ml_accuracy'], 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, 
                yticklabels=label_names,
                ax=axes[0])
    axes[0].set_title('ML Model Accuracy\n(Actual vs Predicted)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # ML Success vs PnL Confusion Matrix
    sns.heatmap(confusion_matrices['ml_success_vs_pnl'], 
                annot=True, fmt='d', cmap='Greens',
                xticklabels=['Negative PnL', 'Positive PnL'], 
                yticklabels=['ML Failed', 'ML Success'],
                ax=axes[1])
    axes[1].set_title('ML Success vs PnL')
    axes[1].set_xlabel('PnL')
    axes[1].set_ylabel('ML Success')
    
    # TP/SL vs PnL Confusion Matrix
    sns.heatmap(confusion_matrices['tp_sl_vs_pnl'], 
                annot=True, fmt='d', cmap='Reds',
                xticklabels=['Negative PnL', 'Positive PnL'], 
                yticklabels=['No TP/SL', 'TP/SL Hit'],
                ax=axes[2])
    axes[2].set_title('TP/SL Hit vs PnL')
    axes[2].set_xlabel('PnL')
    axes[2].set_ylabel('TP/SL Status')
    
    plt.tight_layout()
    
    if save_path:
        print(f"Saving confusion matrix plot to: {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Plot saved successfully!")
    
    # Don't show the plot interactively to avoid hanging
    plt.close()  # Close the figure to free memory

def analyze_trades(file_path, output_path='C:\\Users\\lenovo\\Downloads\\Framework\\shap_feat__USDCAD_model_output.csv'):
    print(f"Loading data from: {file_path}")
    # Load data
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows of data")
    
    # Print column names to debug
    print(f"Available columns: {list(df.columns)}")
    
    # Check if required columns exist
    required_columns = ['y_actual', 'y_predict', 'pnl', 'tp_sl_status']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        print("Available columns:", list(df.columns))
        return None, None, None

    # Fill NaN in tp_sl_status with "No TP/SL"
    df['tp_sl_status'] = df['tp_sl_status'].fillna('No TP/SL')

    # Ensure pnl is numeric and convert if needed
    if df['pnl'].dtype == object:
        # Map common string forms to numeric
        df['pnl'] = df['pnl'].map({'+ve': 1, '-ve': -1, '1':1, '-1':-1}).fillna(0).astype(int)

    # Ensure y_actual and y_predict are numeric
    df['y_actual'] = pd.to_numeric(df['y_actual'], errors='coerce')
    df['y_predict'] = pd.to_numeric(df['y_predict'], errors='coerce')
    
    # Remove rows with NaN values in prediction columns
    df = df.dropna(subset=['y_actual', 'y_predict'])
    print(f"After removing NaN values: {len(df)} rows")

    # Analyze the distribution of values
    print(f"\n=== VALUE DISTRIBUTION ANALYSIS ===")
    print(f"y_actual unique values: {sorted(df['y_actual'].unique())}")
    print(f"y_predict unique values: {sorted(df['y_predict'].unique())}")
    
    # Count occurrences of each value
    print(f"\ny_actual value counts:")
    print(df['y_actual'].value_counts().sort_index())
    print(f"\ny_predict value counts:")
    print(df['y_predict'].value_counts().sort_index())
    
    # Check if 0 values exist and what they might represent
    zero_actual = (df['y_actual'] == 0).sum()
    zero_predict = (df['y_predict'] == 0).sum()
    
    if zero_actual > 0 or zero_predict > 0:
        print(f"\n=== ZERO VALUE ANALYSIS ===")
        print(f"Rows where y_actual = 0: {zero_actual}")
        print(f"Rows where y_predict = 0: {zero_predict}")
        print(f"Zero values might represent:")
        print(f"  - Neutral/no signal predictions")
        print(f"  - Missing or invalid data")
        print(f"  - Hold/no action signals")
        
        # Show sample of rows with zero values
        zero_rows = df[(df['y_actual'] == 0) | (df['y_predict'] == 0)]
        if not zero_rows.empty:
            print(f"\nSample of rows with zero values:")
            print(zero_rows[['y_actual', 'y_predict', 'pnl', 'tp_sl_status']].head())

    print("Applying categorization...")
    # Apply categorization
    df['Category'] = df.apply(categorize_trade, axis=1)

    # Count categories
    counts = df['Category'].value_counts().reset_index()
    counts.columns = ['Category', 'Count']

    print("Creating confusion matrices...")
    # Create confusion matrices
    confusion_matrices = create_confusion_matrices(df)
    
    if confusion_matrices:
        # Plot confusion matrices
        plot_confusion_matrices(confusion_matrices, 
                              save_path='C:\\Users\\lenovo\\Downloads\\Framework\\confusion_matrices.png')

    print(f"Saving categorized data to: {output_path}")
    # Save categorized dataframe
    df.to_csv(output_path, index=False)

    print("Category counts:")
    print(counts)
    
    print("\nConfusion Matrix Analysis:")
    print("=" * 50)
    
    # ML Accuracy metrics
    y_actual = df['y_actual']
    y_pred = df['y_predict']
    ml_accuracy = (y_actual == y_pred).mean()
    print(f"ML Model Accuracy: {ml_accuracy:.2%}")
    
    # PnL analysis
    positive_pnl_rate = (df['pnl'] > 0).mean()
    print(f"Positive PnL Rate: {positive_pnl_rate:.2%}")
    
    # TP/SL analysis
    tp_sl_hit_rate = (df['tp_sl_status'].str.lower() != 'no tp/sl').mean()
    print(f"TP/SL Hit Rate: {tp_sl_hit_rate:.2%}")

    # Classification report for y_actual vs y_pred
    if confusion_matrices and 'labels' in confusion_matrices:
        print("\nClassification Report (y_actual vs y_pred):")
        print(classification_report(y_actual, y_pred, labels=confusion_matrices['labels'], zero_division=0))

    print("Analysis completed successfully!")
    return df, counts, confusion_matrices

if __name__ == "__main__":
    # Example usage:
    df, counts, confusion_matrices = analyze_trades(r'C:\Users\lenovo\Downloads\Framework\shap_feat\USDCAD_tearsheet_final_master_sequence.csv')
