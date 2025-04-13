import os
import pandas as pd
from config import TICKERS, START_DATE, END_DATE
from data_collection import fetch_stock_data
from macro_data import fetch_macro_indicators
from data_preprocessing import preprocess_data
from prophet_model import train_prophet_model
import time

def run_pipeline(ticker, ts):
    try:
        if not ticker.isalpha() or len(ticker) > 5:
            raise ValueError(f"Invalid ticker format: {ticker}")

        os.makedirs('static', exist_ok=True)

        stock_data = fetch_stock_data(ticker)
        if stock_data.empty:
            raise RuntimeError(f"No data found for {ticker}")
            
        stock_csv = f"static/{ticker}_raw_data.csv"
        stock_data.to_csv(stock_csv, index=False)
        print(f"Stock data saved to {stock_csv}")

        macro_data = fetch_macro_indicators()
        if macro_data is None or macro_data.empty:
            raise RuntimeError("Failed to fetch macroeconomic data")
            
        macro_csv = "static/macro_indicators.csv"
        macro_data.to_csv(macro_csv, index=False)
        print(f"Macro data saved to {macro_csv}")

        processed_data = preprocess_data(stock_data, macro_data)
        if processed_data.empty:
            raise RuntimeError("Data preprocessing failed")
            
        processed_csv = f"static/{ticker}_processed.csv"
        processed_data.to_csv(processed_csv, index=False)
        print(f"Processed data saved to {processed_csv}")

        model, forecast, report_path = train_prophet_model(processed_data, ticker, timestamp=ts)
        if not os.path.exists(report_path):
            raise RuntimeError(f"Failed to generate forecast plot at {report_path}")
            
        print(f"Prophet training complete. Plot: {report_path}")

        return model, forecast, report_path

    except Exception as e:
        print(f"Pipeline failure: {str(e)}")
        for f in [stock_csv, macro_csv, processed_csv]:
            if os.path.exists(f):
                os.remove(f)
        raise

if __name__ == "__main__":
    for ticker in TICKERS:
        try:
            run_pipeline(ticker, str(int(time.time())))
        except Exception as e:
            print(f"Failed processing {ticker}: {str(e)}")
            continue
        print(f"Pipeline completed successfully for {ticker}.")
# This script is designed to be run as a standalone module. It will execute the pipeline for each ticker in TICKERS.
