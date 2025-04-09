import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_percentage_error, 
    r2_score
)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from prophet import Prophet

def plot_predictions(test_dates, y_test, y_pred, ticker):
    plt.figure(figsize=(16, 8))
    dates = test_dates.to_numpy()
    actual = y_test.to_numpy()
    predicted = y_pred

    # Main plot lines
    plt.plot(dates, actual, label='Actual Prices', color='#1f77b4', linewidth=2.5, alpha=0.9)
    plt.plot(dates, predicted, label='Model Predictions', color='#ff7f0e', linestyle='--', linewidth=2, alpha=0.9)
    
    # Uncertainty bands
    residuals = actual - predicted
    volatility = pd.Series(residuals).rolling(14).std().dropna()
    if len(volatility) > 0:
        valid_dates = dates[-len(volatility):]
        plt.fill_between(valid_dates,
                         predicted[-len(volatility):] - 1.96*volatility,
                         predicted[-len(volatility):] + 1.96*volatility,
                         color='gray', alpha=0.2, label='95% Confidence Band')

    plt.title(f'{ticker} Price Predictions\nHybrid Model Performance', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    plt.grid(alpha=0.2)
    plt.tight_layout()
    
    plot_path = os.path.join('static', f'{ticker}_hybrid_pred_{int(time.time())}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def train_model(data, ticker='STOCK'):
    """Hybrid Prophet + Gradient Boosting model with enhanced features"""
    # Feature Engineering
    required_features = [
        'SP500_Relative', 'Volatility_7', 'Volatility_30',
        'Close_Lag_7', 'Close_Lag_14', 'Interest_Rate_MA30',
        'Momentum_14', 'Market_Cap_Relative'
    ]
    
    # Add technical features if missing
    if 'SP500_Relative' not in data.columns:
        data = add_technical_indicators(data)
    
    # Split data with time-based validation
    split_date = data['Date'].iloc[int(len(data)*0.8)]
    train = data[data['Date'] < split_date]
    test = data[data['Date'] >= split_date]

    # Stage 1: Prophet for trend/seasonality
    prophet_model, forecast = train_prophet_model(data)
    train['Prophet_Pred'] = forecast['yhat'][:len(train)]
    
    # Stage 2: Gradient Boosting for residuals
    train['Residual'] = train['Close'] - train['Prophet_Pred']
    
    model = Pipeline([
        ('scaler', RobustScaler()),
        ('regressor', GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.8,
            validation_fraction=0.2,
            n_iter_no_change=15,
            random_state=42
        ))
    ])
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    X_train = train[required_features]
    y_train = train['Residual']
    
    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=tscv, 
        scoring='neg_mean_absolute_percentage_error'
    )
    print(f"Cross-Val MAPE: {-cv_scores.mean():.2%} (±{cv_scores.std():.2%})")
    
    # Final training
    model.fit(X_train, y_train)
    
    # Prepare test data
    test['Prophet_Pred'] = forecast['yhat'][len(train):len(train)+len(test)]
    X_test = test[required_features]
    test['Residual_Pred'] = model.predict(X_test)
    test['Hybrid_Pred'] = test['Prophet_Pred'] + test['Residual_Pred']
    
    # Evaluation
    y_test = test['Close'].to_numpy()
    y_pred = test['Hybrid_Pred'].to_numpy()
    
    print("\n=== Model Performance ===")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred):.2%}")
    print(f"R²: {r2_score(y_test, y_pred):.2f}")
    
    # Feature Importance
    importances = model.named_steps['regressor'].feature_importances_
    print("\nFeature Importance Ranking:")
    for feat, imp in sorted(zip(required_features, importances), key=lambda x: x[1], reverse=True):
        print(f"{feat:.<25} {imp:.4f}")
    
    # Plotting
    plot_path = plot_predictions(test['Date'], y_test, y_pred, ticker)
    
    return model, X_test, y_test, y_pred, test['Date'].to_numpy(), plot_path

# Helper functions
def add_technical_indicators(data):
    """Enhanced feature engineering"""
    df = data.copy()
    
    # Price momentum
    df['Momentum_14'] = df['Close'].pct_change(14)
    
    # Volatility measures
    df['Volatility_7'] = df['Close'].pct_change().rolling(7).std()
    df['Volatility_30'] = df['Close'].pct_change().rolling(30).std()
    
    # Relative market performance
    df['SP500_Relative'] = df['Close'] / df['SP500']
    
    # Lagged prices
    for lag in [7, 14]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    
    # Market cap proxy
    df['Market_Cap_Relative'] = (df['Close'] * df['Volume']) / df['SP500']
    
    return df.dropna()

def train_prophet_model(data):
    """Prophet model with price constraints"""
    df = data[['Date', 'Close', 'SP500', 'Interest_Rate']].copy()
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Price constraints
    max_price = df['y'].max() * 2
    df['cap'] = max_price
    df['floor'] = 0
    
    model = Prophet(
        growth='logistic',
        changepoint_prior_scale=0.05,
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=False
    )
    
    # Add market regressors
    model.add_regressor('SP500')
    model.add_regressor('Interest_Rate')
    
    model.fit(df)
    
    future = model.make_future_dataframe(periods=30)
    future['cap'] = max_price
    future['floor'] = 0
    future['SP500'] = df['SP500'].ffill().values[-1]
    future['Interest_Rate'] = df['Interest_Rate'].ffill().values[-1]
    
    forecast = model.predict(future)
    
    # Apply price floor
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    
    return model, forecast