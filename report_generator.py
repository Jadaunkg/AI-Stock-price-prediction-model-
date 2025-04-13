from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot
import pandas as pd
import os

def create_full_report(ticker, actual_data, forecast_data, historical_data, ts, aggregation=None):
    """
    Create a full HTML report with interactive charts and tables.
    
    Parameters:
      ticker (str): The stock ticker.
      actual_data (pd.DataFrame): Aggregated historical actuals. Must contain either a 'YearMonth' or 'Period' column.
      forecast_data (pd.DataFrame): Aggregated forecast data. Must contain either a 'YearMonth' or 'Period' column.
      historical_data (pd.DataFrame): The raw historical price data.
      ts: Timestamp used for the report filename.
      aggregation (str, optional): Additional info regarding aggregation (unused in logic, but available if needed).
    """
    # Determine which time column to use for the aggregated forecast and actual data.
    if 'YearMonth' in forecast_data.columns:
        forecast_time_col = 'YearMonth'
    elif 'Period' in forecast_data.columns:
        forecast_time_col = 'Period'
    else:
        raise ValueError("Forecast data must have a 'YearMonth' or 'Period' column.")
        
    if 'YearMonth' in actual_data.columns:
        actual_time_col = 'YearMonth'
    elif 'Period' in actual_data.columns:
        actual_time_col = 'Period'
    else:
        raise ValueError("Actual data must have a 'YearMonth' or 'Period' column.")    
    
    # Use a period label: commonly "Month" for 'YearMonth' or "Period" if custom.
    period_label = "Month" if forecast_time_col == "YearMonth" else "Period"
    
    colors = {
        'actual': '#1f77b4',
        'Low': '#d62728',
        'Average': '#2ca02c',
        'High': '#9467bd'
    }
    
    # ---- Forecast Chart ----
    forecast_chart_fig = go.Figure()
    forecast_chart_fig.add_trace(go.Scatter(
        x=actual_data[actual_time_col],
        y=actual_data['Average'],
        mode='lines+markers',
        name='Actual Price',
        line=dict(color=colors['actual'], width=2.5),
        marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br><b>Price</b>: %{y:.2f}<extra></extra>"
    ))
    
    base_price = actual_data['Average'].iloc[-1]
    for col in ['Low', 'Average', 'High']:
        forecast_chart_fig.add_trace(go.Scatter(
            x=forecast_data[forecast_time_col],
            y=forecast_data[col],
            mode='lines+markers',
            name='Forecast ' + col,
            line=dict(color=colors[col], width=2.5),
            marker=dict(size=8, line=dict(width=1, color='#ffffff')),
            hovertemplate="<b>%{x}</b><br><b>" + col + "</b>: %{y:.2f}<extra></extra>"
        ))
        final_value = forecast_data[col].iloc[-1]
        pct_change = ((final_value - base_price) / base_price) * 100
        annotation_text = (
            "<b>{:.2f}</b><br><span style='color:{};'>{:+.1f}%</span>"
            .format(final_value, colors[col], pct_change)
        )
        forecast_chart_fig.add_annotation(
            x=forecast_data[forecast_time_col].iloc[-1],
            y=final_value,
            text=annotation_text,
            showarrow=True,
            arrowhead=2,
            ax=20,
            ay=-40,
            bgcolor='white',
            font=dict(color=colors[col], size=10),
            bordercolor='black',
            borderwidth=1
        )
    forecast_chart_fig.update_layout(
        title=ticker + " Price Forecast",
        xaxis_title=period_label,
        yaxis_title="Price",
        template="plotly_white"
    )
    
    # ---- Forecast Table ----
    forecast_table_fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{period_label}</b>", "<b>Low</b>", "<b>Average</b>", "<b>High</b>"],
            fill_color=colors['High'],
            align='center',
            font=dict(color='white', size=12),
            height=40
        ),
        cells=dict(
            values=[
                forecast_data[forecast_time_col],
                forecast_data['Low'].round(2),
                forecast_data['Average'].round(2),
                forecast_data['High'].round(2)
            ],
            fill_color='white',
            align='center',
            font=dict(color='black', size=12),
            height=35,
            format=['', '.2f', '.2f', '.2f']
        )
    )])
    forecast_table_fig.update_layout(
        title=ticker + " Forecast Details",
        template="plotly_white"
    )
    
    # ---- Historical Chart ----
    historical_chart_fig = go.Figure()
    historical_chart_fig.add_trace(go.Scatter(
        x=historical_data['Date'],
        y=historical_data['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#9b59b6', width=1.8),
        hovertemplate="<b>Date</b>: %{x|%d %b %Y}<br><b>Price</b>: %{y:.2f}<extra></extra>"
    ))
    historical_chart_fig.update_layout(
        title=ticker + " Historical Price Analysis",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis_title="Price",
        template="plotly_white"
    )
    
    # ---- Historical Table (Monthly Aggregation) ----
    historical_monthly = historical_data.resample('M', on='Date').agg({
        'Close': ['min', 'max', 'mean']
    }).reset_index()
    historical_monthly.columns = ['Date', 'Low', 'High', 'Average']
    historical_monthly['YearMonth'] = historical_monthly['Date'].dt.strftime('%Y-%m')
    historical_table_fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Month</b>", "<b>Low</b>", "<b>High</b>", "<b>Average</b>"],
            fill_color='#2c3e50',
            align='center',
            font=dict(color='white', size=12),
            height=40
        ),
        cells=dict(
            values=[
                historical_monthly['YearMonth'],
                historical_monthly['Low'].round(2),
                historical_monthly['High'].round(2),
                historical_monthly['Average'].round(2)
            ],
            fill_color='white',
            align='center',
            font=dict(color='black', size=12),
            height=35,
            format=['', '.2f', '.2f', '.2f']
        )
    )])
    historical_table_fig.update_layout(
        title=ticker + " Historical Price Statistics",
        template="plotly_white"
    )
    
    # ---- Generate HTML Components ----
    forecast_chart_html = plot(forecast_chart_fig, output_type='div', include_plotlyjs='cdn')
    forecast_table_html = plot(forecast_table_fig, output_type='div', include_plotlyjs=False)
    historical_chart_html = plot(historical_chart_fig, output_type='div', include_plotlyjs=False)
    historical_table_html = plot(historical_table_fig, output_type='div', include_plotlyjs=False)
    
    custom_style = """
    <style>
      body { margin: 0; padding: 0; }
      .report-container {
          max-width: 1200px;
          margin: 2rem auto;
          padding: 1rem;
          background: #ffffff;
          box-shadow: 0 12px 24px rgba(0,0,0,0.1);
          border-radius: 16px;
          border: 1px solid rgba(0,0,0,0.1);
      }
      .report-title {
          text-align: center;
          color: #2c3e50;
          margin-bottom: 1rem;
          font-size: 2rem;
          font-weight: 600;
      }
      .section {
          margin-bottom: 2rem;
      }
    </style>
    """
    
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{ticker} Stock Analysis Report</title>
    {custom_style}
</head>
<body>
    <div class="report-container">
        <h1 class="report-title">{ticker} Stock Forecast & Historical Analysis</h1>
        <div class="section">
            {forecast_chart_html}
        </div>
        <div class="section">
            {forecast_table_html}
        </div>
        <div class="section">
            {historical_chart_html}
        </div>
        <div class="section">
            {historical_table_html}
        </div>
    </div>
</body>
</html>
"""
    
    report_filename = f"{ticker}_professional_report_{ts}.html"
    report_path = os.path.join('static', report_filename)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    return report_path
