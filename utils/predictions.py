import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import streamlit as st

class AdvancedPredictions:
    @staticmethod
    def add_time_features(df):
        """Add time-based features for better predictions."""
        df = df.copy()
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        return df

    @staticmethod
    def prepare_features(df):
        """Prepare features for model training."""
        df = AdvancedPredictions.add_time_features(df)
        feature_columns = ['day_of_week', 'month', 'quarter', 'year', 
                         'day_of_month', 'week_of_year']
        return df[feature_columns], df['revenue']

    @staticmethod
    def train_models(X, y):
        """Train multiple models and return their predictions."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        }

        # Train models and collect metrics
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results[name] = {
                'model': model,
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'mape': mean_absolute_percentage_error(y_test, y_pred),
                'test_predictions': y_pred,
                'test_actual': y_test,
                'test_indices': X_test.index
            }

        return results, X_test.index[-1]

    @staticmethod
    def generate_future_features(last_date, prediction_days):
        """Generate feature matrix for future predictions."""
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                   periods=prediction_days)
        future_df = pd.DataFrame({'date': future_dates})
        future_df = AdvancedPredictions.add_time_features(future_df)
        return future_df.drop('date', axis=1)

    @staticmethod
    def predict_trends(df, prediction_days):
        """Generate advanced sales predictions with multiple models."""
        try:
            # Prepare daily data
            daily_sales = df.groupby('date')['revenue'].sum().reset_index()

            # Prepare features and target
            X, y = AdvancedPredictions.prepare_features(daily_sales)

            # Train models and get results
            model_results, last_idx = AdvancedPredictions.train_models(X, y)

            # Generate future features
            future_features = AdvancedPredictions.generate_future_features(
                daily_sales['date'].iloc[-1], prediction_days
            )

            # Make predictions
            predictions = {}
            for name, results in model_results.items():
                future_pred = results['model'].predict(future_features)
                predictions[name] = future_pred

            # Create visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Model Predictions Comparison',
                    'Model Performance Metrics',
                    'Prediction Error Distribution',
                    'Feature Importance (Random Forest)'
                ]
            )

            # Historical data and predictions
            colors = {'Linear Regression': '#1f77b4', 
                     'Random Forest': '#2ca02c', 
                     'XGBoost': '#ff7f0e'}

            # Plot 1: Model Predictions
            fig.add_trace(
                go.Scatter(
                    x=daily_sales['date'],
                    y=daily_sales['revenue'],
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='#666666', width=2)
                ),
                row=1, col=1
            )

            future_dates = pd.date_range(
                start=daily_sales['date'].iloc[-1] + pd.Timedelta(days=1),
                periods=prediction_days
            )

            for name, pred in predictions.items():
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=pred,
                        mode='lines',
                        name=f'{name} Prediction',
                        line=dict(color=colors[name], dash='dash')
                    ),
                    row=1, col=1
                )

            # Plot 2: Model Performance Metrics
            metrics_data = pd.DataFrame({
                'Model': list(model_results.keys()),
                'RMSE': [results['rmse'] for results in model_results.values()],
                'R²': [results['r2'] for results in model_results.values()],
                'MAPE': [results['mape'] * 100 for results in model_results.values()]
            })

            for metric in ['RMSE', 'R²', 'MAPE']:
                fig.add_trace(
                    go.Bar(
                        name=metric,
                        x=metrics_data['Model'],
                        y=metrics_data[metric],
                        text=metrics_data[metric].round(3),
                        textposition='auto',
                    ),
                    row=1, col=2
                )

            # Plot 3: Error Distribution
            for name, results in model_results.items():
                errors = results['test_actual'] - results['test_predictions']
                fig.add_trace(
                    go.Histogram(
                        x=errors,
                        name=name,
                        opacity=0.7,
                        nbinsx=30
                    ),
                    row=2, col=1
                )

            # Plot 4: Feature Importance (Random Forest)
            rf_model = model_results['Random Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)

            fig.add_trace(
                go.Bar(
                    x=feature_importance['importance'],
                    y=feature_importance['feature'],
                    orientation='h',
                    name='Feature Importance'
                ),
                row=2, col=2
            )

            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white',
                title={
                    'text': 'Advanced Sales Prediction Analysis',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )

            # Calculate ensemble prediction (weighted average based on R² scores)
            weights = np.array([results['r2'] for results in model_results.values()])
            weights = weights / weights.sum()
            ensemble_predictions = np.average(
                np.column_stack(list(predictions.values())),
                weights=weights,
                axis=1
            )

            return fig, ensemble_predictions, model_results

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None, None, None
