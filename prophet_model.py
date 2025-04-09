from prophet import Prophet
import pandas as pd
from report_generator import create_full_report

def train_prophet_model(data, ticker='STOCK', periods=365):
    # For a specific ticker, adjust historic Close values.
    if ticker == 'TSLA':
        split_date = pd.to_datetime('2020-08-31')
        data.loc[data['Date'] < split_date, 'Close'] *= 5

    max_price = data['Close'].max() * 2
    data['cap'] = max_price
    data['floor'] = 0
    
    model = Prophet(
        growth='logistic',
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_mode='multiplicative',
        uncertainty_samples=50
    )
    for feature in ['RSI', 'MACD', 'Interest_Rate']:
        if feature in data.columns:
            model.add_regressor(feature)
    df = data.rename(columns={'Date': 'ds', 'Close': 'y'})
    df[['cap', 'floor']] = [max_price, 0]
    model.fit(df)
    
    future = model.make_future_dataframe(periods=periods)
    future[['cap', 'floor']] = [max_price, 0]
    for feature in ['RSI', 'MACD', 'Interest_Rate']:
        if feature in data.columns:
            future[feature] = df[feature].values[-1]
    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
    
    # Compute actual monthly averages for the last 8 months.
    last_date = df['ds'].max()
    eight_months_ago = last_date - pd.DateOffset(months=8)
    actual_recent = df[df['ds'] >= eight_months_ago].copy()
    actual_recent['YearMonth'] = actual_recent['ds'].dt.to_period('M').dt.strftime('%Y-%m')
    actual_monthly = actual_recent.groupby('YearMonth').agg({'y': 'mean'}).reset_index()
    actual_monthly.rename(columns={'y': 'Average'}, inplace=True)
    actual_monthly = actual_monthly.sort_values('YearMonth')
    
    # Compute forecast monthly predictions for the next 12 months.
    forecast_future = forecast[forecast['ds'] >= last_date].copy()
    forecast_future['YearMonth'] = forecast_future['ds'].dt.to_period('M').dt.strftime('%Y-%m')
    forecast_monthly = forecast_future.groupby('YearMonth').agg({
        'yhat_lower': 'min',
        'yhat': 'mean',
        'yhat_upper': 'max'
    }).reset_index()
    forecast_monthly.rename(columns={
        'yhat_lower': 'Low',
        'yhat': 'Average',
        'yhat_upper': 'High'
    }, inplace=True)
    forecast_monthly = forecast_monthly.sort_values('YearMonth').head(12)
    last_actual_value = actual_monthly['Average'].iloc[-1]
    forecast_monthly.loc[forecast_monthly.index[0], ['Low', 'Average', 'High']] = last_actual_value
    
    historical_data = data.copy()
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    
    report_path = create_full_report(
        ticker=ticker,
        actual_monthly=actual_monthly,
        forecast_monthly=forecast_monthly,
        historical_data=historical_data
    )
    
    return model, forecast, report_path
