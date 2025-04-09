from datetime import datetime
# config.py
# List of tickers can be a single ticker or a list if you want to iterate over multiple stocks.
TICKERS = ["TSLA"]  # e.g., you can later add: ["AAPL", "GOOGL", "MSFT"]

# Date range for data collection
START_DATE = "1900-01-01"  # Earliest practical date
END_DATE = datetime.today().strftime("%Y-%m-%d")  # Current date
