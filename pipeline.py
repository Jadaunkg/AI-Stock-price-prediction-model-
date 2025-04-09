# pipeline.py
import pandas as pd
from config import TICKERS, START_DATE, END_DATE
from data_collection import fetch_stock_data
from macro_data import fetch_macro_indicators
from data_preprocessing import preprocess_data
from prophet_model import train_prophet_model  # Updated import

def run_pipeline(ticker):
    # Fetch stock data
    stock_data = fetch_stock_data(ticker)
    stock_csv = f"{ticker}_raw_data.csv"
    stock_data.to_csv(stock_csv, index=False)
    print(f"Stock data saved to {stock_csv}")

    # Fetch macroeconomic data
    macro_data = fetch_macro_indicators()
    macro_csv = "macro_indicators.csv"
    if macro_data is not None:
        macro_data.to_csv(macro_csv, index=False)
        print(f"Macro data saved to {macro_csv}")
    else:
        print("Macro data could not be fetched.")

    # Preprocess data
    processed_data = preprocess_data(stock_data, macro_data)
    processed_csv = f"{ticker}_processed.csv"
    processed_data.to_csv(processed_csv, index=False)
    print(f"Processed data saved to {processed_csv}")

    # Train Prophet model (forecast along with report generation)
    model, forecast, plot_path = train_prophet_model(processed_data, ticker)
    print(f"Training complete! Forecast plot saved to {plot_path}")

if __name__ == "__main__":
    for ticker in TICKERS:
        run_pipeline(ticker)
