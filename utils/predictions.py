import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import streamlit as st # Assuming streamlit is used for dark/light mode


def predict_trends(df, prediction_days):
    """Generate sales trend predictions with confidence intervals."""
    try:
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

        # Calculate confidence intervals
        confidence = 0.95
        n = len(X)
        m = len(future_days)
        dof = n - 2
        mse = np.sum((y - model.predict(X)) ** 2) / dof

        # Standard error of prediction
        x_mean = np.mean(X)
        x_std = np.std(X)

        std_errors = np.sqrt(mse * (1 + 1/n + (future_days - x_mean)**2 / (n * x_std**2)))
        t_value = stats.t.ppf((1 + confidence) / 2, dof)

        ci_lower = predictions - t_value * std_errors
        ci_upper = predictions + t_value * std_errors

        # Create visualization
        fig = make_subplots(rows=1, cols=1, subplot_titles=['Sales Trend Prediction'])

        # Historical data
        fig.add_trace(
            go.Scatter(
                x=daily_sales['date'],
                y=daily_sales['revenue'],
                mode='lines',
                name='Historical Sales',
                line=dict(color='#0080ff', width=2)
            )
        )

        # Predictions
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines',
                name='Predicted Sales',
                line=dict(color='#00ff00', width=2, dash='dash')
            )
        )

        # Confidence intervals
        fig.add_trace(
            go.Scatter(
                x=future_dates.tolist() + future_dates.tolist()[::-1],
                y=ci_upper.tolist() + ci_lower.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,128,255,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{int(confidence*100)}% Confidence Interval'
            )
        )

        fig.update_layout(
            title={
                'text': "Sales Trend Prediction",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            showlegend=True,
            template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white',
            height=600,
            hovermode='x unified'
        )

        return fig, predictions, ci_lower, ci_upper

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None, None, None

def generate_recommendations(df):
    """Generate product recommendations based on sales patterns."""
    try:
        # Calculate product metrics
        product_metrics = df.groupby('product_id').agg({
            'revenue': ['sum', 'mean', 'std'],
            'quantity': ['sum', 'mean', 'std']
        }).reset_index()

        # Flatten column names
        product_metrics.columns = ['product_id', 'total_revenue', 'avg_revenue', 'std_revenue',
                                'total_quantity', 'avg_quantity', 'std_quantity']

        # Calculate rankings
        product_metrics['revenue_rank'] = product_metrics['total_revenue'].rank(ascending=False)
        product_metrics['quantity_rank'] = product_metrics['total_quantity'].rank(ascending=False)
        product_metrics['stability_rank'] = (product_metrics['std_revenue'] / product_metrics['avg_revenue']).rank()

        # Calculate composite score (lower is better)
        product_metrics['composite_score'] = (
            0.4 * product_metrics['revenue_rank'] + 
            0.4 * product_metrics['quantity_rank'] + 
            0.2 * product_metrics['stability_rank']
        )

        # Get top 5 recommendations with detailed metrics
        recommendations = product_metrics.nsmallest(5, 'composite_score')[
            ['product_id', 'total_revenue', 'avg_revenue', 'total_quantity', 'composite_score']
        ]

        # Format the metrics
        recommendations['total_revenue'] = recommendations['total_revenue'].round(2)
        recommendations['avg_revenue'] = recommendations['avg_revenue'].round(2)
        recommendations['composite_score'] = recommendations['composite_score'].round(3)

        return recommendations

    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return None