import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker):
    """Fetch and process maximum historical stock data for a single ticker"""
    # Fetch data with maximum history
    data = yf.download(
        ticker,
        period="max",
        auto_adjust=True,
        progress=False
    )
    
    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    data = data.reset_index()
    
    # Clean column names (handle both single and multi-ticker cases)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
        data.columns = [col.split('_')[0] for col in data.columns]
    else:
        data.columns = [col.split('_')[0] for col in data.columns]

    #Standardize Column name
    column_map = {
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    }
    data = data.rename(columns=column_map)

    # Handle date column

    date_cols = [col for col in data.columns if 'date' in str(col).lower()]
    if date_cols:
        data = data.rename(columns={date_cols[0]: 'Date'})
    
    # Convert and validate dates
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce', utc=True).dt.tz_localize(None)
    data = data.dropna(subset=['Date']).sort_values('Date')

    # Validate required columns
    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Data quality checks
    if len(data) < 100:
        raise ValueError(f"Need ≥100 data points (got {len(data)})")

    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
    if data['Volume'].mean() < 900:
        raise ValueError(f"Low liquidity (avg volume: {data['Volume'].mean():.0f})")

    return data[required]

#Ensuring DataFrame has a properly formatted “Date” column
import pandas as pd
from feature_engineering import add_technical_indicators

def enforce_date_column(df, df_name):
    """Standardize date column across datasets"""
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{df_name} data is not a DataFrame")
    
    # Clean column names similarly to fetch_stock_data
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns]
        df.columns = [col.split('_')[0] for col in df.columns]
    
    # Find and rename date column
    date_cols = [col for col in df.columns if 'date' in str(col).lower()]
    if date_cols:
        df = df.rename(columns={date_cols[0]: 'Date'})
    elif 'Date' not in df.columns:
        raise ValueError(f"{df_name} data missing Date column")
    
    # Convert and validate dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
    df['Date'] = df['Date'].dt.tz_localize(None)
    df = df.dropna(subset=['Date']).sort_values('Date')
    
    return df

#Preprocessing the data

def preprocess_data(stock_df, macro_df):
    """Merge and align stock data with macroeconomic data"""
    # Standardize both datasets
    stock = enforce_date_column(stock_df.copy(), "Stock")
    macro = enforce_date_column(macro_df.copy(), "Macro")
    
    stock = stock.rename(columns={
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    })

    #Date Alignment
    start_date = max(stock['Date'].min(), macro['Date'].min())
    end_date = min(stock['Date'].max(), macro['Date'].max())
    
    # Create full date range

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Reindex and forward-fill missing values
    stock_processed = (
        stock.set_index('Date')
        .reindex(date_range)
        .rename_axis('Date')
        .ffill()
        .bfill()
        .reset_index()
    )
    
    macro_processed = (
        macro.set_index('Date')
        .reindex(date_range)
        .rename_axis('Date')
        .ffill()
        .bfill()
        .reset_index()
    )

    #merge datasets
    merged = pd.merge_asof(
    stock_processed.sort_values('Date'),
    macro_processed.sort_values('Date'),
    on='Date',
    direction='nearest'
)
    
    #including macro Features
    # Define required macroeconomic features that should be present after merging
    # These typically include processed features like moving averages and volatility metrics
    required_macro = [
        'Volatility_14', 'Momentum_7', 'Price_Diff',
        'Interest_Rate_MA30', 'SP500_MA30'
    ]

    #checking missing macro
    missing_macro = [col for col in required_macro if col not in merged.columns]
    if missing_macro:
        print(f"Warning: Missing macro features {missing_macro}, using defaults")
        for col in missing_macro:
            merged[col] = 2.5 if col == 'Interest_Rate' else 4000

    # Ensure minimum data points for technical indicator calculations
    if len(merged) < 30:
        raise ValueError("Not enough data to compute technical indicators. Minimum 30 rows required.")
    
    # Verify presence of essential raw macroeconomic features
    required_macro_features = ['Interest_Rate', 'SP500']
    missing_macro = [f for f in required_macro_features if f not in merged.columns]
    if missing_macro:
        raise ValueError(f"Missing macro features: {missing_macro}")
    
    # Add technical indicators using external function
    merged = add_technical_indicators(merged)
    
    # Verify presence of essential stock market data columns
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_columns if col not in merged.columns]
    if missing:
        raise ValueError(f"Missing columns after merging: {missing}")
    
# Final Feature Selection

# Define complete set of features for model input
    required_output_features = [
    # Core price data
    'Date', 'Close',
    
    # Time-based feature
    'Days',            # Trading days count
    
    # Macroeconomic features (raw and processed)
    'Interest_Rate',      # Current interest rate
    'SP500',             # S&P 500 index value
    'Interest_Rate_MA30', # 30-day MA of interest rates
    'SP500_MA30'        # 30-day MA of S&P 500
]
    
    # Return final dataset with curated features
    return merged[required_output_features]
