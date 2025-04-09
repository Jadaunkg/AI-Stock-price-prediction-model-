from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot
import pandas as pd
import os
import time

def create_full_report(ticker, actual_monthly, forecast_monthly, historical_data):
    # Define a clear color palette
    colors = {
        'actual': '#1f77b4',      # blue for actual price
        'Low': '#d62728',         # red for low forecast
        'Average': '#2ca02c',     # green for average forecast
        'High': '#9467bd'         # purple for high forecast
    }
    
    # -------------------------------
    # Build Forecast Chart (monthly data)
    # -------------------------------
    forecast_chart_fig = go.Figure()

    # Plot the actual price line (monthly)
    forecast_chart_fig.add_trace(go.Scatter(
        x=actual_monthly['YearMonth'],
        y=actual_monthly['Average'],
        mode='lines+markers',
        name='Actual Price',
        line=dict(color=colors['actual'], width=2.5),
        marker=dict(size=8),
        hovertemplate="<b>Month</b>: %{x}<br><b>Price</b>: %{y:.2f}<extra></extra>"
    ))
    
    base_price = actual_monthly['Average'].iloc[-1]
    # Plot forecast lines (Low, Average, High)
    for col in ['Low', 'Average', 'High']:
        forecast_chart_fig.add_trace(go.Scatter(
            x=forecast_monthly['YearMonth'],
            y=forecast_monthly[col],
            mode='lines+markers',
            name='Forecast ' + col,
            line=dict(color=colors[col], width=2.5),
            marker=dict(size=8, line=dict(width=1, color='#ffffff')),
            hovertemplate="<b>Month</b>: %{x}<br><b>" + col + "</b>: %{y:.2f}<extra></extra>"
        ))
        # Add an annotation at the end of each forecast line
        final_value = forecast_monthly[col].iloc[-1]
        pct_change = ((final_value - base_price) / base_price) * 100
        annotation_text = (
            "<b>{:.2f}</b><br><span style='color:{};'>{:+.1f}%</span>"
            .format(final_value, colors[col], pct_change)
        )
        forecast_chart_fig.add_annotation(
            x=forecast_monthly['YearMonth'].iloc[-1],
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
        xaxis_title="Month",
        yaxis_title="Price",
        template="plotly_white"
    )
    
    # -------------------------------
    # Build Forecast Table
    # -------------------------------
    forecast_table_fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Month</b>", "<b>Low</b>", "<b>Average</b>", "<b>High</b>"],
            fill_color=colors['High'],
            align='center',
            font=dict(color='white', size=12),
            height=40
        ),
        cells=dict(
            values=[
                forecast_monthly['YearMonth'],
                forecast_monthly['Low'].round(2),
                forecast_monthly['Average'].round(2),
                forecast_monthly['High'].round(2)
            ],
            fill_color='white',
            align='center',
            font=dict(color='black', size=12),
            height=35,
            format=['', '.2f', '.2f', '.2f']
        )
    )])
    forecast_table_fig.update_layout(
        title=ticker + " 12-Month Forecast Details",
        template="plotly_white"
    )
    
    # -------------------------------
    # Build Historical Chart (daily data with range selectors)
    # -------------------------------
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
    
    # -------------------------------
    # Build Historical Table (monthly aggregates)
    # -------------------------------
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
    
    # -------------------------------
    # Generate HTML snippets (for each figure)
    # -------------------------------
    # Only include Plotly JS once.
    forecast_chart_html = plot(forecast_chart_fig, output_type='div', include_plotlyjs='cdn')
    forecast_table_html = plot(forecast_table_fig, output_type='div', include_plotlyjs=False)
    historical_chart_html = plot(historical_chart_fig, output_type='div', include_plotlyjs=False)
    historical_table_html = plot(historical_table_fig, output_type='div', include_plotlyjs=False)
    
    # -------------------------------
    # Build the final HTML which can be embedded via an iframe.
    # The embed snippet (iframe) is placed at the bottom as a comment for convenience.
    # -------------------------------
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
    
    # Assemble the final HTML
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
        <!--
        Use the following iframe code to embed this report in your WordPress post:
        <iframe src="YOUR_DOMAIN/{ticker}_professional_report_{int(time.time())}.html" style="width: 100%; height: 1000px;" loading="lazy" title="{ticker} Stock Analysis Report" frameborder="0"></iframe>
        -->
    </div>
</body>
</html>
"""
    
    # Save the report HTML in the "static" folder
    report_filename = f"{ticker}_professional_report_{int(time.time())}.html"
    report_path = os.path.join('static', report_filename)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    return report_path
