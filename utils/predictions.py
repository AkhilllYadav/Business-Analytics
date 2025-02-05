import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

def predict_trends(df, prediction_days):
    """Generate sales trend predictions."""
    # Prepare time series data
    daily_sales = df.groupby('date')['revenue'].sum().reset_index()
    daily_sales['days_from_start'] = (daily_sales['date'] - daily_sales['date'].min()).dt.days
    
    # Prepare features and target
    X = daily_sales['days_from_start'].values.reshape(-1, 1)
    y = daily_sales['revenue'].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate future dates
    last_date = daily_sales['date'].max()
    future_dates = pd.date_range(start=last_date, periods=prediction_days + 1)[1:]
    future_days = np.array(range(
        daily_sales['days_from_start'].max() + 1,
        daily_sales['days_from_start'].max() + prediction_days + 1
    )).reshape(-1, 1)
    
    # Make predictions
    predictions = model.predict(future_days)
    
    # Create visualization
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=daily_sales['date'],
        y=daily_sales['revenue'],
        mode='lines',
        name='Historical Sales'
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines',
        name='Predicted Sales',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(
        title="Sales Trend Prediction",
        xaxis_title="Date",
        yaxis_title="Revenue",
        showlegend=True
    )
    
    return fig

def generate_recommendations(df):
    """Generate product recommendations based on sales patterns."""
    # Calculate product metrics
    product_metrics = df.groupby('product_id').agg({
        'revenue': ['sum', 'mean'],
        'quantity': ['sum', 'mean']
    }).reset_index()
    
    # Flatten column names
    product_metrics.columns = ['product_id', 'total_revenue', 'avg_revenue',
                             'total_quantity', 'avg_quantity']
    
    # Calculate rankings
    product_metrics['revenue_rank'] = product_metrics['total_revenue'].rank(ascending=False)
    product_metrics['quantity_rank'] = product_metrics['total_quantity'].rank(ascending=False)
    
    # Calculate composite score
    product_metrics['composite_score'] = (
        product_metrics['revenue_rank'] + product_metrics['quantity_rank']
    ) / 2
    
    # Get top 5 recommendations
    recommendations = product_metrics.nsmallest(5, 'composite_score')[
        ['product_id', 'total_revenue', 'total_quantity']
    ]
    
    return recommendations
