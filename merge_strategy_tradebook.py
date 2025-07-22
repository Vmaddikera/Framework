import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def load_and_prepare_data(strat_log_path: str, tradebook_path: str):
    """
    Load and prepare the strategy log and tradebook data.
    
    Args:
        strat_log_path (str): Path to strategy log CSV
        tradebook_path (str): Path to tradebook CSV
        
    Returns:
        tuple: (strat_log_df, tradebook_df)
    """
    print("Loading strategy log...")
    strat_log_df = pd.read_csv(strat_log_path)
    print(f"Strategy log loaded: {len(strat_log_df)} rows")
    print(f"Columns: {list(strat_log_df.columns)}")
    
    print("\nLoading tradebook...")
    tradebook_df = pd.read_csv(tradebook_path)
    print(f"Tradebook loaded: {len(tradebook_df)} rows")
    print(f"Columns: {list(tradebook_df.columns)}")
    
    return strat_log_df, tradebook_df

def convert_timestamps(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Convert timestamp column to datetime format.
    
    Args:
        df (pd.DataFrame): DataFrame with timestamp column
        timestamp_col (str): Name of timestamp column
        
    Returns:
        pd.DataFrame: DataFrame with converted timestamps
    """
    df = df.copy()
    
    # Try different timestamp formats
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    except:
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='%Y-%m-%d %H:%M:%S')
        except:
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='%Y-%m-%d %H:%M:%S.%f')
            except:
                print(f"Warning: Could not parse timestamps in {timestamp_col}")
    
    return df

def match_trades_to_strategy_sequence(strat_log_df: pd.DataFrame, tradebook_df: pd.DataFrame, 
                                    gap_threshold_hours: int = 6) -> pd.DataFrame:
    """
    Match tradebook data to strategy log based on strategy start_datetime sequence.
    Fill gaps with the same trade information.
    
    Args:
        strat_log_df (pd.DataFrame): Strategy log data
        tradebook_df (pd.DataFrame): Tradebook data
        gap_threshold_hours (int): Hours threshold to consider as a gap
        
    Returns:
        pd.DataFrame: Merged DataFrame with strategy and tradebook data
    """
    print("\nMatching trades to strategy sequence...")
    
    # Convert timestamps
    strat_log_df = convert_timestamps(strat_log_df, 'timestamp')
    tradebook_df = convert_timestamps(tradebook_df, 'EntryTime')
    tradebook_df = convert_timestamps(tradebook_df, 'ExitTime')
    
    # Sort strategy log by timestamp
    strat_log_df = strat_log_df.sort_values('timestamp').reset_index(drop=True)
    
    # Sort tradebook by entry time
    tradebook_df = tradebook_df.sort_values('EntryTime').reset_index(drop=True)
    
    print(f"Strategy log entries: {len(strat_log_df)}")
    print(f"Trades in tradebook: {len(tradebook_df)}")
    
    # Initialize result DataFrame with strategy log structure
    result_data = []
    current_trade_idx = 0
    
    for idx, strat_row in strat_log_df.iterrows():
        strat_timestamp = strat_row['timestamp']
        
        # Find the current trade that covers this strategy timestamp
        current_trade = None
        if current_trade_idx < len(tradebook_df):
            current_trade = tradebook_df.iloc[current_trade_idx]
        
        # Check if we need to move to next trade
        if current_trade is not None:
            # If strategy timestamp is after current trade's exit time, move to next trade
            if strat_timestamp > current_trade['ExitTime']:
                current_trade_idx += 1
                if current_trade_idx < len(tradebook_df):
                    current_trade = tradebook_df.iloc[current_trade_idx]
                else:
                    current_trade = None
        
        # Create combined row with all strategy columns (keeping original names)
        combined_row = {}
        
        # Add all strategy columns with their original names
        for col in strat_log_df.columns:
            combined_row[col] = strat_row[col]
        
        # Add trade data if available
        if current_trade is not None and strat_timestamp >= current_trade['EntryTime'] and strat_timestamp <= current_trade['ExitTime']:
            # Trade is active during this strategy timestamp
            
            # Convert PnL to binary (1 for positive, -1 for negative)
            pnl_value = current_trade.get('PnL', 0)
            if isinstance(pnl_value, (int, float)):
                pnl_binary = 1 if pnl_value > 0 else (-1 if pnl_value < 0 else 0)
            else:
                pnl_binary = 0
            
            # Get tp_SL status from tradebook Reason column
            tp_sl_status = current_trade.get('Reason', '')
            
            combined_row.update({
                'pnl': pnl_binary,
                'return_pct': current_trade.get('ReturnPct', ''),
                'duration': current_trade.get('Duration', ''),
                'tp_sl_status': tp_sl_status
            })
        else:
            # No active trade for this strategy timestamp
            combined_row.update({
                'pnl': 0,
                'return_pct': '',
                'duration': '',
                'tp_sl_status': ''
            })
        
        result_data.append(combined_row)
        
        # Progress indicator
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(strat_log_df)} strategy entries, current trade: {current_trade_idx}")
    
    result_df = pd.DataFrame(result_data)
    
    # Fill gaps with previous trade information
    print("\nFilling gaps with previous trade information...")
    result_df = fill_trade_gaps(result_df, gap_threshold_hours)
    
    print(f"\n Final result: {len(result_df)} rows")
    print(f" Active trades: {len(result_df[result_df['pnl'] != 0])}")
    print(f" Strategy entries: {len(strat_log_df)}")
    
    return result_df

def fill_trade_gaps(df: pd.DataFrame, gap_threshold_hours: int = 6) -> pd.DataFrame:
    """
    Fill gaps in trade information with previous trade data.
    
    Args:
        df (pd.DataFrame): DataFrame with trade and strategy data
        gap_threshold_hours (int): Hours threshold to consider as a gap
        
    Returns:
        pd.DataFrame: DataFrame with filled gaps
    """
    df = df.copy()
    
    # Convert timestamps for gap detection
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Initialize variables
    last_trade_data = None
    gap_start_idx = None
    
    for idx, row in df.iterrows():
        if row['pnl'] != 0:  # Active trade (non-zero PnL)
            # Active trade found, update last trade data
            last_trade_data = {
                'pnl': row['pnl'],
                'return_pct': row['return_pct'],
                'duration': row['duration'],
                'tp_sl_status': row.get('tp_sl_status', ''),
                'ExitTime': row['timestamp']  # Use current timestamp as exit time for gap calculation
            }
            gap_start_idx = None
        else:
            # No active trade, check if we should fill with previous trade data
            if last_trade_data is not None:
                # Check if this is within gap threshold from last trade
                if gap_start_idx is None:
                    gap_start_idx = idx
                
                # Calculate time difference from last trade
                last_trade_time = last_trade_data['ExitTime']
                current_time = row['timestamp']
                time_diff = current_time - last_trade_time
                
                # If within gap threshold, fill with last trade data
                if time_diff <= pd.Timedelta(hours=gap_threshold_hours):
                    df.loc[idx, 'pnl'] = last_trade_data['pnl']
                    df.loc[idx, 'return_pct'] = last_trade_data['return_pct']
                    df.loc[idx, 'duration'] = last_trade_data['duration']
                    df.loc[idx, 'tp_sl_status'] = last_trade_data['tp_sl_status']
    
    return df

def analyze_missing_trades(strat_log_df: pd.DataFrame, tradebook_df: pd.DataFrame, 
                          time_tolerance_seconds: int = 60):
    """
    Analyze which trades are missing and why.
    
    Args:
        strat_log_df (pd.DataFrame): Strategy log data
        tradebook_df (pd.DataFrame): Tradebook data
        time_tolerance_seconds (int): Time tolerance for matching
    """
    print("\n=== Analyzing Missing Trades ===")
    print(f"Total trades in tradebook: {len(tradebook_df)}")
    print(f"Total strategy entries: {len(strat_log_df)}")
    print(f"Time tolerance: {time_tolerance_seconds} seconds")
    print("-" * 50)
    
    # Convert timestamps
    strat_log_df = convert_timestamps(strat_log_df, 'timestamp')
    tradebook_df = convert_timestamps(tradebook_df, 'EntryTime')
    tradebook_df = convert_timestamps(tradebook_df, 'ExitTime')
    
    # Sort data
    strat_log_df = strat_log_df.sort_values('timestamp').reset_index(drop=True)
    tradebook_df = tradebook_df.sort_values('EntryTime').reset_index(drop=True)
    
    # Analyze each trade
    matched_trades = []
    unmatched_trades = []
    
    for idx, trade in tradebook_df.iterrows():
        entry_time = trade['EntryTime']
        exit_time = trade['ExitTime']
        
        # Find strategy entries within tolerance of entry time
        time_diff = abs(strat_log_df['timestamp'] - entry_time)
        matching_strat_entries = strat_log_df[time_diff <= pd.Timedelta(seconds=time_tolerance_seconds)]
        
        if len(matching_strat_entries) > 0:
            # Take the closest match
            closest_match_idx = time_diff[time_diff <= pd.Timedelta(seconds=time_tolerance_seconds)].idxmin()
            strat_entry = strat_log_df.loc[closest_match_idx]
            closest_time_diff = time_diff[closest_match_idx]
            
            matched_trades.append({
                'trade_index': idx,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'PnL': trade.get('PnL', ''),
                'ReturnPct': trade.get('ReturnPct', ''),
                'Duration': trade.get('Duration', ''),
                'matched_strategy_time': strat_entry['timestamp'],
                'time_difference_seconds': closest_time_diff.total_seconds(),
                'matched_strategy_row': closest_match_idx
            })
        else:
            # Find the closest strategy entry (even if outside tolerance)
            min_time_diff_idx = time_diff.idxmin()
            min_time_diff = time_diff[min_time_diff_idx]
            closest_strat_time = strat_log_df.loc[min_time_diff_idx, 'timestamp']
            
            unmatched_trades.append({
                'trade_index': idx,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'PnL': trade.get('PnL', ''),
                'ReturnPct': trade.get('ReturnPct', ''),
                'Duration': trade.get('Duration', ''),
                'closest_strategy_time': closest_strat_time,
                'time_difference_seconds': min_time_diff.total_seconds(),
                'closest_strategy_row': min_time_diff_idx
            })
    
    # Create DataFrames for analysis
    matched_df = pd.DataFrame(matched_trades)
    unmatched_df = pd.DataFrame(unmatched_trades)
    
    print(f"\n Matched trades: {len(matched_df)}")
    print(f" Unmatched trades: {len(unmatched_df)}")
    
    if len(unmatched_df) > 0:
        print(f"\n=== DETAILED ANALYSIS OF MISSING TRADES ===")
        print(f"The following {len(unmatched_df)} trades could not be matched:")
        print("-" * 80)
        
        for idx, trade in unmatched_df.iterrows():
            print(f"\nTrade #{trade['trade_index'] + 1}:")
            print(f"  Entry Time: {trade['entry_time']}")
            print(f"  Exit Time:  {trade['exit_time']}")
            print(f"  PnL: {trade['PnL']}")
            print(f"  Return: {trade['ReturnPct']}%")
            print(f"  Duration: {trade['Duration']}")
            print(f"  Closest Strategy Time: {trade['closest_strategy_time']}")
            print(f"  Time Difference: {trade['time_difference_seconds']:.1f} seconds ({trade['time_difference_seconds']/60:.1f} minutes)")
            print(f"  Strategy Row: {trade['closest_strategy_row']}")
            
            # Check if there are any strategy entries in the trade duration
            trade_duration_entries = strat_log_df[
                (strat_log_df['timestamp'] >= trade['entry_time']) & 
                (strat_log_df['timestamp'] <= trade['exit_time'])
            ]
            
            if len(trade_duration_entries) > 0:
                print(f"   Found {len(trade_duration_entries)} strategy entries during trade duration")
                print(f"    Strategy entries: {list(trade_duration_entries['timestamp'])}")
            else:
                print(f"   No strategy entries found during trade duration")
        
        # Summary statistics
        print(f"\n=== SUMMARY STATISTICS ===")
        print(f"Average time difference for unmatched trades: {unmatched_df['time_difference_seconds'].mean():.1f} seconds")
        print(f"Maximum time difference: {unmatched_df['time_difference_seconds'].max():.1f} seconds")
        print(f"Minimum time difference: {unmatched_df['time_difference_seconds'].min():.1f} seconds")
        
        # Save detailed analysis
        unmatched_df.to_csv('missing_trades_analysis.csv', index=False)
        print(f"\n✓ Detailed analysis saved to: missing_trades_analysis.csv")
    
    if len(matched_df) > 0:
        print(f"\n=== MATCHED TRADES STATISTICS ===")
        print(f"Average time difference for matched trades: {matched_df['time_difference_seconds'].mean():.1f} seconds")
        print(f"Maximum time difference: {matched_df['time_difference_seconds'].max():.1f} seconds")
        print(f"Minimum time difference: {matched_df['time_difference_seconds'].min():.1f} seconds")
        
        # Save matched trades analysis
        matched_df.to_csv('matched_trades_analysis.csv', index=False)
        print(f"✓ Matched trades analysis saved to: matched_trades_analysis.csv")
    
    return matched_df, unmatched_df

def try_different_tolerances(strat_log_df: pd.DataFrame, tradebook_df: pd.DataFrame):
    """
    Try different time tolerances to see how many trades can be matched.
    """
    print(f"\n=== TESTING DIFFERENT TIME TOLERANCES ===")
    
    tolerances = [30, 60, 120, 300, 600, 1800]  # 30s, 1min, 2min, 5min, 10min, 30min
    
    for tolerance in tolerances:
        print(f"\nTesting tolerance: {tolerance} seconds ({tolerance/60:.1f} minutes)")
        
        # Convert timestamps
        strat_log_df = convert_timestamps(strat_log_df, 'timestamp')
        tradebook_df = convert_timestamps(tradebook_df, 'EntryTime')
        
        # Count matches
        matched_count = 0
        for idx, trade in tradebook_df.iterrows():
            entry_time = trade['EntryTime']
            time_diff = abs(strat_log_df['timestamp'] - entry_time)
            matching_entries = strat_log_df[time_diff <= pd.Timedelta(seconds=tolerance)]
            
            if len(matching_entries) > 0:
                matched_count += 1
        
        print(f"  Matched trades: {matched_count}/{len(tradebook_df)} ({matched_count/len(tradebook_df)*100:.1f}%)")

def create_final_master_csv(strat_log_path: str, tradebook_path: str, output_path: str = None, gap_threshold_hours: int = 6):
    """
    Create final master CSV by matching tradebook to strategy sequence.
    
    Args:
        strat_log_path (str): Path to strategy log CSV
        tradebook_path (str): Path to tradebook CSV
        output_path (str): Output path for final CSV (optional)
        gap_threshold_hours (int): Hours threshold for gap filling
    """
    print("=== Creating Final Master CSV (Strategy Sequence Based) ===")
    print(f"Strategy Log: {strat_log_path}")
    print(f"Tradebook: {tradebook_path}")
    print(f"Gap threshold: {gap_threshold_hours} hours")
    print("-" * 50)
    
    # Load data
    strat_log_df, tradebook_df = load_and_prepare_data(strat_log_path, tradebook_path)
    
    # Match and merge data
    final_df = match_trades_to_strategy_sequence(strat_log_df, tradebook_df, gap_threshold_hours)
    
    if not final_df.empty:
        # Set output path
        if output_path is None:
            output_path = tradebook_path.replace('_tradebook.csv', '_final_master_sequence.csv')
        
        # Save the full detailed CSV with all columns
        final_df.to_csv(output_path, index=False)
        print(f"\nFinal master CSV saved to: {output_path}")
        print(f" Total rows: {len(final_df)}")
        print(f" Active trade rows: {len(final_df[final_df['pnl'] != 0])}")
        
        # Count strategy columns (excluding trade columns)
        strategy_cols = [col for col in final_df.columns if col not in ['pnl', 'return_pct', 'duration', 'tp_sl_status']]
        print(f"Strategy columns: {len(strategy_cols)}")
        print(f" Trade columns: pnl, return_pct, duration, tp_sl_status")
        
        # Show sample data
        print("\nSample of final data:")
        print(final_df.head(10))
        
        # # Create simplified version with only the 4 required trade columns
        # simplified_output_path = output_path.replace('_final_master_sequence.csv', '_simplified_4columns.csv')
        # simplified_df = final_df[['pnl', 'return_pct', 'duration', 'tp_sl_status']].copy()
        # simplified_df.to_csv(simplified_output_path, index=False)
        # print(f"\nSimplified 4-column CSV saved to: {simplified_output_path}")
        
        return final_df
    else:
        print("No data to save!")
        return pd.DataFrame()

def main():
    """
    Main function to run the sequence-based merge process with missing trades analysis.
    """
    # File paths
    strat_log_path = r"C:\Users\lenovo\Downloads\Framework\weighted_feat\strategy_log_weighted.csv"
    tradebook_path = r"C:\Users\lenovo\Downloads\Framework\weighted_feat\USDCAD_tearsheet_tradebook.csv"
    
    # Load data for analysis
    strat_log_df, tradebook_df = load_and_prepare_data(strat_log_path, tradebook_path)
    
    print(f"\nExpected total trades: {len(tradebook_df)}")
    
    # STEP 1: Analyze missing trades first
    print("\n" + "="*60)
    print("STEP 1: ANALYZING MISSING TRADES")
    print("="*60)
    
    matched_df, unmatched_df = analyze_missing_trades(strat_log_df, tradebook_df, time_tolerance_seconds=60)
    
    # Test different tolerances
    try_different_tolerances(strat_log_df, tradebook_df)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total trades: {len(tradebook_df)}")
    print(f"Successfully matched: {len(matched_df)}")
    print(f"Missing trades: {len(unmatched_df)}")
    print(f"Match rate: {len(matched_df)/len(tradebook_df)*100:.1f}%")
    
    # STEP 2: Create final master CSV with sequence-based matching
    print("\n" + "="*60)
    print("STEP 2: CREATING FINAL MASTER CSV")
    print("="*60)
    
    final_df = create_final_master_csv(strat_log_path, tradebook_path, gap_threshold_hours=6)
    
    if not final_df.empty:
        print("\n=== Final Summary ===")
        print(f"Total strategy entries: {len(final_df)}")
        print(f"Active trade entries: {len(final_df[final_df['pnl'] != 0])}")
        
        # Show trade statistics
        active_trades = final_df[final_df['pnl'] != 0]
        if not active_trades.empty:
            print(f"\nTrade Statistics:")
            
            # Show PnL binary distribution
            pnl_counts = active_trades['pnl'].value_counts()
            print(f"  - PnL Distribution:")
            print(f"     Positive trades (1): {pnl_counts.get(1, 0)}")
            print(f"     Negative trades (-1): {pnl_counts.get(-1, 0)}")
            print(f"    Neutral trades (0): {pnl_counts.get(0, 0)}")
            
            # Show tp_SL status distribution
            if 'tp_sl_status' in active_trades.columns:
                tp_sl_counts = active_trades['tp_sl_status'].value_counts()
                print(f"  - TP/SL Status Distribution:")
                for status, count in tp_sl_counts.items():
                    if pd.notna(status) and status != '':
                        print(f"    * {status}: {count}")
        
        print(f"\n=== FILES CREATED ===")
        print(f" Full detailed CSV: USDCAD_tearsheet_final_master_sequence.csv")
        # print(f" Simplified 4-column CSV: USDCAD_tearsheet_simplified_4columns.csv")
        print(f" Missing trades analysis: missing_trades_analysis.csv")
        print(f" Matched trades analysis: matched_trades_analysis.csv")
        
        # Show column information
        print(f"\n=== COLUMN INFORMATION ===")
        strategy_cols = [col for col in final_df.columns if col not in ['pnl', 'return_pct', 'duration', 'tp_sl_status']]
        trade_cols = ['pnl', 'return_pct', 'duration', 'tp_sl_status']
        
        print(f"Strategy columns ({len(strategy_cols)}): {strategy_cols[:5]}{'...' if len(strategy_cols) > 5 else ''}")
        print(f"Trade columns ({len(trade_cols)}): {trade_cols}")
        
    else:
        print("No data to process!")

if __name__ == "__main__":
    main()