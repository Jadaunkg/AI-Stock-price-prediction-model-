import yfinance as yf
import pandas as pd
import logging
from datetime import datetime

# Configure logging to display info and error messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_stock_data(ticker, start_date=None, end_date=None):
    
    # Validate the date range
    if start_date and end_date:
        if pd.to_datetime(start_date) > pd.to_datetime(end_date):
            raise ValueError("start_date must be before end_date")
    
    # If no dates are provided, use "max" to fetch the entire available period.
    period = "max" if (start_date is None and end_date is None) else None

    logger.info(f"Fetching data for ticker: {ticker} from {start_date or 'the beginning'} to {end_date or 'today'}")
    
    # Download data without the group_by parameter to avoid multi-index issues for single tickers.
    data = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        period=period,
        auto_adjust=True,    # Adjust prices for splits/dividends.
        progress=False       # Disable progress bar.
    )
    
    # Check if data is empty
    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    
    # Reset the index so that the date becomes a column
    data = data.reset_index()
    
    # Debug: Log the fetched columns
    logger.info(f"Fetched columns: {list(data.columns)}")
    
    # Flatten columns:
    # If the first column is a tuple, then the columns are MultiIndex style.
    if isinstance(data.columns[0], tuple):
        data.columns = [col[0] for col in data.columns]
    else:
        # If columns are strings, remove any suffixes separated by underscores.
        data.columns = [col.split('_')[0] if isinstance(col, str) else col for col in data.columns]
    
    # Rename the date column to 'Date' if needed.
    date_cols = [col for col in data.columns if 'date' in col.lower()]
    if date_cols:
        data = data.rename(columns={date_cols[0]: 'Date'})
    
    # Convert the 'Date' column to datetime objects
    try:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce', utc=True).dt.tz_localize(None)
        invalid_dates = data['Date'].isna().sum()
    except Exception as e:
        logger.error(f"Error processing dates: {e}")
        raise

    if invalid_dates > 0:
        logger.warning(f"Found {invalid_dates} invalid dates; these rows will be dropped.")
    data = data.dropna(subset=['Date']).sort_values('Date')
    
    # Define required columns
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # Check for missing columns and log available columns for debugging
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"Data columns available: {list(data.columns)}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert the 'Volume' column to numeric (if necessary)
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
    
    return data[required_columns]

# Example usage
if __name__ == "__main__":
    try:
        # For maximum available data, do not pass start_date or end_date.
        ticker_symbol = "TSLA"  # Replace with any ticker symbol as needed.
        stock_data = fetch_stock_data(ticker_symbol)
        logger.info(f"Fetched {len(stock_data)} rows of data for {ticker_symbol}.")
        # Print the first few rows of the fetched data.
        print(stock_data.head())
    except Exception as e:
        logger.error(f"An error occurred: {e}")
