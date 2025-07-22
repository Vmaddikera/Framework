import re
from datetime import datetime
from io import StringIO
import pandas as pd
import concurrent.futures
import time
from selectolax.parser import HTMLParser
from pathlib import Path

class TearsheetExtractor:
    """
    Class to extract relevant data from HTML tearsheet files:
     - Tradebook DataFrame
     - Equity DataFrame
     - Stock name, timeframe, start date, end date
    """
    expected_tradebook_cols = {
        'EntryBar', 'ExitBar', 'Ticker', 'Size', 'EntryPrice', 'ExitPrice', 'PnL',
        'ReturnPct', 'EntryTime', 'ExitTime', 'Duration', 'Reason'
    }
    
    tp_sl_related_cols = [
        'tp_SL', 'tp_sl', 'tpSL', 'tpsl', 'TP_SL', 'TP_SL_status', 'Status',
        'ExitReason', 'Exit_Reason', 'Reason', 'Exit Type', 'ExitType'
    ]

    @staticmethod
    def extract_tradebook_from_html(html_content: str) -> pd.DataFrame:
        """
        Extract tradebook DataFrame from HTML content.
        """
        try:
            tables = pd.read_html(StringIO(html_content))
        except Exception:
            return pd.DataFrame()
        
        for df in tables:
            if TearsheetExtractor.expected_tradebook_cols.issubset(set(df.columns)):
                cleaned_df = TearsheetExtractor.clean_tradebook(df)
                
                # Check for tp_SL related columns and add them if found
                available_tp_sl_cols = []
                for col in TearsheetExtractor.tp_sl_related_cols:
                    if col in df.columns:
                        available_tp_sl_cols.append(col)
                        cleaned_df[col] = df[col]
                
                if available_tp_sl_cols:
                    print(f"Found tp_SL related columns: {available_tp_sl_cols}")
                
                return cleaned_df
        return pd.DataFrame()

    @staticmethod
    def clean_tradebook(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and convert tradebook columns to correct types.
        """
        df = df.copy()
        df['EntryTime'] = pd.to_datetime(df['EntryTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df['ExitTime'] = pd.to_datetime(df['ExitTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        return df

    @staticmethod
    def extract_stock_and_timeframe(html_content: str) -> tuple:
        """
        Extract stock name and timeframe from the <h4> tag in tearsheet HTML.
        Returns:
            (stock_name, timeframe) or (None, None)
        """
        tree = HTMLParser(html_content)
        h4 = tree.css_first('h4')
        if not h4:
            return None, None
        text = h4.text(separator=" ").strip()
        stock_match = re.search(r'Stock:\s*([^|]+)', text)
        timeframe_match = re.search(r'Timeframe:\s*([^\s|]+)', text)
        stock_name = stock_match.group(1).strip() if stock_match else None
        timeframe = timeframe_match.group(1).strip() if timeframe_match else None
        return stock_name, timeframe

    @staticmethod
    def extract_ticker_info(html_content: str) -> dict:
        """
        Extract stock name, timeframe, and start/end dates from tearsheet HTML.
        Returns:
            dict with keys 'stock_name', 'timeframe', 'start_date', 'end_date', 'exchange'
            Dates as datetime.date objects or None
        """
        tree = HTMLParser(html_content)

        stock_name = None
        timeframe = None
        exchange = None

        h4 = tree.css_first('h4')
        if h4:
            text = h4.text(separator=" ").strip()
            stock_match = re.search(r'Stock:\s*([^|]+)', text)
            timeframe_match = re.search(r'Timeframe:\s*([^\s|]+)', text)
            exchange_match = re.search(r'Exchange:\s*([^\s|]+)', text)
            if stock_match:
                stock_name = stock_match.group(1).strip()
            if timeframe_match:
                timeframe = timeframe_match.group(1).strip()
            if exchange_match:
                exchange = exchange_match.group(1).strip()

        # Extract date range from <h1><dt>
        start_date = None
        end_date = None

        dt_tag = tree.css_first('h1 dt')
        if dt_tag and dt_tag.text():
            date_range_text = dt_tag.text().strip()
            # Format example: "4 Jan, 2005 - 13 Jun, 2024"
            parts = date_range_text.split('-')
            if len(parts) == 2:
                try:
                    start_date = datetime.strptime(parts[0].strip(), "%d %b, %Y").date()
                    end_date = datetime.strptime(parts[1].strip(), "%d %b, %Y").date()
                except ValueError:
                    # Could attempt different parsing or return None
                    pass

        return {
            'stock_name': stock_name,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'exchange': exchange
        }

    @staticmethod
    def _extract_single(html_content: str, filename: str) -> dict:
        tables = []
        try:
            tables = pd.read_html(StringIO(html_content))
        except Exception:
            pass

        # Extract tradebook from tables
        tradebook = pd.DataFrame()
        for df in tables:
            if TearsheetExtractor.expected_tradebook_cols.issubset(set(df.columns)):
                tradebook = TearsheetExtractor.clean_tradebook(df)
                
                # Add tp_SL related columns if found
                for col in TearsheetExtractor.tp_sl_related_cols:
                    if col in df.columns:
                        tradebook[col] = df[col]
                break

        # Extract equity from tables
        equity = None
        for df in tables:
            cols_lower = [str(col).lower() for col in df.columns]
            if any(k in col for col in cols_lower for k in ('equity', 'asset', 'cash')):
                equity = df
                break

        ticker_info = TearsheetExtractor.extract_ticker_info(html_content)

        return {
            'tradebook': tradebook,
            'equity': equity,
            'stock_name': ticker_info.get('stock_name'),
            'timeframe': ticker_info.get('timeframe'),
            'start_date': ticker_info.get('start_date'),
            'end_date': ticker_info.get('end_date'),
            'exchange': ticker_info.get('exchange'),
            'filename': filename
        }

    @staticmethod
    def extract_all(html_contents: list, filenames: list, max_workers: int = 4) -> list:
        """
        Extract data from multiple HTML strings concurrently using multiprocessing.
        """
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(TearsheetExtractor._extract_single, html_contents, filenames))
        return results

def extract_tradebook_from_file(html_file_path: str) -> pd.DataFrame:
    """
    Extract tradebook DataFrame from an HTML tearsheet file.
    
    Args:
        html_file_path (str): Path to the HTML tearsheet file
        
    Returns:
        pd.DataFrame: Tradebook data or empty DataFrame if extraction fails
    """
    try:
        # Read the HTML file
        with open(html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        # Extract tradebook using the TearsheetExtractor
        tradebook_df = TearsheetExtractor.extract_tradebook_from_html(html_content)
        
        if not tradebook_df.empty:
            print(f"Successfully extracted tradebook with {len(tradebook_df)} trades")
            print(f"Columns: {list(tradebook_df.columns)}")
            return tradebook_df
        else:
            print("No tradebook data found in the HTML file")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error reading file {html_file_path}: {e}")
        return pd.DataFrame()

def extract_tradebook_with_info(html_file_path: str) -> dict:
    """
    Extract tradebook and additional information from HTML tearsheet.
    
    Args:
        html_file_path (str): Path to the HTML tearsheet file
        
    Returns:
        dict: Dictionary containing tradebook DataFrame and ticker information
    """
    try:
        # Read the HTML file
        with open(html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        # Extract tradebook
        tradebook_df = TearsheetExtractor.extract_tradebook_from_html(html_content)
        
        # Extract ticker information
        ticker_info = TearsheetExtractor.extract_ticker_info(html_content)
        
        result = {
            'tradebook': tradebook_df,
            'stock_name': ticker_info.get('stock_name'),
            'timeframe': ticker_info.get('timeframe'),
            'start_date': ticker_info.get('start_date'),
            'end_date': ticker_info.get('end_date'),
            'exchange': ticker_info.get('exchange'),
            'filename': Path(html_file_path).name
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing file {html_file_path}: {e}")
        return {}

def process_tearsheet_file(html_file_path: str, save_csv: bool = True) -> dict:
    """
    Process a single tearsheet HTML file and extract all relevant data.
    
    Args:
        html_file_path (str): Path to the HTML tearsheet file
        save_csv (bool): Whether to save tradebook to CSV
        
    Returns:
        dict: Dictionary containing all extracted data
    """
    print(f"Processing tearsheet: {html_file_path}")
    print("-" * 50)
    
    # Extract tradebook and info
    result = extract_tradebook_with_info(html_file_path)
    
    if result and not result['tradebook'].empty:
        tradebook_df = result['tradebook']
        
        # print(f" Stock: {result['stock_name']}")
        # print(f" Timeframe: {result['timeframe']}")
        # print(f" Exchange: {result['exchange']}")
        # print(f" Date Range: {result['start_date']} to {result['end_date']}")
        # print(f" Number of trades: {len(tradebook_df)}")
        
        # Show all available columns
        print(f"\nAvailable columns in tradebook:")
        for i, col in enumerate(tradebook_df.columns, 1):
            print(f"  {i}. {col}")
        
        # Check for tp_SL related columns
        found_tp_sl_cols = []
        for col in TearsheetExtractor.tp_sl_related_cols:
            if col in tradebook_df.columns:
                found_tp_sl_cols.append(col)
        
        if found_tp_sl_cols:
            print(f"\n Found tp_SL related columns: {found_tp_sl_cols}")
            # Show tp_SL status distribution
            for col in found_tp_sl_cols:
                status_counts = tradebook_df[col].value_counts()
                print(f"\n{col} distribution:")
                for status, count in status_counts.items():
                    if pd.notna(status) and status != '':
                        print(f"  * {status}: {count}")
        else:
            print(f"\n No tp_SL related columns found in tradebook")
            print(f"  Searched for: {TearsheetExtractor.tp_sl_related_cols}")
        
        # Show sample data
        print("\nFirst few trades:")
        print(tradebook_df.head())
        
        # Save to CSV if requested
        if save_csv:
            output_csv = html_file_path.replace('.html', '_tradebook.csv')
            tradebook_df.to_csv(output_csv, index=False)
            print(f"\n Tradebook saved to: {output_csv}")
        
        # Show trade statistics
        print(f"\nTrade Statistics:")
        print(f"  - Total PnL: {tradebook_df['PnL'].sum():.2f}")
        print(f"  - Average Return: {tradebook_df['ReturnPct'].mean():.2f}%")
        print(f"  - Winning trades: {(tradebook_df['PnL'] > 0).sum()}")
        print(f"  - Losing trades: {(tradebook_df['PnL'] < 0).sum()}")
        
        return result
    else:
        print(" Failed to extract data from tearsheet")
        return {}

def main():
    """
    Main function to process tearsheet files.
    """
    # Example usage - replace with your actual HTML file path
    # html
    html_file_path = r"C:\Users\lenovo\Downloads\Framework\weighted_feat\USDCAD_tearsheet.html"
    print("=== Tearsheet Processor ===")
    print("Combined tearsheet extractor and tradebook processor")
    print("=" * 50)
    
    # Process the tearsheet file
    result = process_tearsheet_file(html_file_path, save_csv=True)
    
    if result:
        print(f"\n Successfully processed tearsheet!")
        print(f" Tradebook extracted with {len(result['tradebook'])} trades")
        print(f" CSV file created for further processing")
    else:
        print(f"\n Failed to process tearsheet file")

if __name__ == "__main__":
    main() 