import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

def create_sales_charts(df, date_range):
    """Create sales analytics visualizations."""
    # Filter data by date range
    mask = (df['date'] >= pd.Timestamp(date_range[0])) & (df['date'] <= pd.Timestamp(date_range[1]))
    df_filtered = df[mask]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Daily Sales Revenue", "Top Products by Revenue",
                       "Sales Quantity Over Time", "Revenue vs Quantity Correlation")
    )
    
    # Daily sales revenue
    daily_sales = df_filtered.groupby('date')['revenue'].sum().reset_index()
    fig.add_trace(
        go.Scatter(x=daily_sales['date'], y=daily_sales['revenue'],
                  mode='lines', name='Daily Revenue'),
        row=1, col=1
    )
    
    # Top products by revenue
    top_products = df_filtered.groupby('product_id')['revenue'].sum().sort_values(ascending=True).tail(10)
    fig.add_trace(
        go.Bar(x=top_products.values, y=top_products.index,
               orientation='h', name='Revenue by Product'),
        row=1, col=2
    )
    
    # Sales quantity over time
    daily_quantity = df_filtered.groupby('date')['quantity'].sum().reset_index()
    fig.add_trace(
        go.Scatter(x=daily_quantity['date'], y=daily_quantity['quantity'],
                  mode='lines', name='Daily Quantity'),
        row=2, col=1
    )
    
    # Revenue vs Quantity correlation
    fig.add_trace(
        go.Scatter(x=df_filtered['quantity'], y=df_filtered['revenue'],
                  mode='markers', name='Revenue vs Quantity'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Sales Analytics Dashboard")
    return fig

def create_marketing_charts(df, date_range):
    """Create marketing analytics visualizations."""
    # Filter data by date range
    mask = (df['date'] >= pd.Timestamp(date_range[0])) & (df['date'] <= pd.Timestamp(date_range[1]))
    df_filtered = df[mask]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Daily Marketing Spend", "Campaign Performance",
                       "Click-through Rate Trend", "Spend vs Clicks Correlation")
    )
    
    # Daily marketing spend
    daily_spend = df_filtered.groupby('date')['spend'].sum().reset_index()
    fig.add_trace(
        go.Scatter(x=daily_spend['date'], y=daily_spend['spend'],
                  mode='lines', name='Daily Spend'),
        row=1, col=1
    )
    
    # Campaign performance
    campaign_metrics = df_filtered.groupby('campaign_id').agg({
        'spend': 'sum',
        'clicks': 'sum',
        'impressions': 'sum'
    }).reset_index()
    campaign_metrics['ctr'] = campaign_metrics['clicks'] / campaign_metrics['impressions']
    
    fig.add_trace(
        go.Bar(x=campaign_metrics['campaign_id'], y=campaign_metrics['ctr'],
               name='CTR by Campaign'),
        row=1, col=2
    )
    
    # Click-through rate trend
    daily_ctr = df_filtered.groupby('date').agg({
        'clicks': 'sum',
        'impressions': 'sum'
    }).reset_index()
    daily_ctr['ctr'] = daily_ctr['clicks'] / daily_ctr['impressions']
    
    fig.add_trace(
        go.Scatter(x=daily_ctr['date'], y=daily_ctr['ctr'],
                  mode='lines', name='Daily CTR'),
        row=2, col=1
    )
    
    # Spend vs Clicks correlation
    fig.add_trace(
        go.Scatter(x=df_filtered['spend'], y=df_filtered['clicks'],
                  mode='markers', name='Spend vs Clicks'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Marketing Analytics Dashboard")
    return fig

def create_review_charts(df, date_range):
    """Create review analytics visualizations."""
    # Filter data by date range
    mask = (df['date'] >= pd.Timestamp(date_range[0])) & (df['date'] <= pd.Timestamp(date_range[1]))
    df_filtered = df[mask]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Rating Distribution", "Average Rating Trend",
                       "Top Products by Rating", "Rating Volume Over Time")
    )
    
    # Rating distribution
    rating_dist = df_filtered['rating'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=rating_dist.index, y=rating_dist.values,
               name='Rating Distribution'),
        row=1, col=1
    )
    
    # Average rating trend
    daily_rating = df_filtered.groupby('date')['rating'].mean().reset_index()
    fig.add_trace(
        go.Scatter(x=daily_rating['date'], y=daily_rating['rating'],
                  mode='lines', name='Avg Rating Trend'),
        row=1, col=2
    )
    
    # Top products by rating
    product_ratings = df_filtered.groupby('product_id')['rating'].agg(['mean', 'count']).reset_index()
    product_ratings = product_ratings[product_ratings['count'] >= 5]  # Min 5 reviews
    top_products = product_ratings.sort_values('mean', ascending=True).tail(10)
    
    fig.add_trace(
        go.Bar(x=top_products['mean'], y=top_products['product_id'],
               orientation='h', name='Avg Rating by Product'),
        row=2, col=1
    )
    
    # Rating volume over time
    daily_volume = df_filtered.groupby('date').size().reset_index(name='count')
    fig.add_trace(
        go.Scatter(x=daily_volume['date'], y=daily_volume['count'],
                  mode='lines', name='Review Volume'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Review Analytics Dashboard")
    return fig
