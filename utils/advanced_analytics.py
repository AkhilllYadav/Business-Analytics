import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.ensemble import IsolationForest
from scipy import stats

class AdvancedAnalytics:
    @staticmethod
    def analyze_sentiment(review_text):
        """Analyze sentiment of review text using TextBlob."""
        analysis = TextBlob(str(review_text))
        # Returns polarity (-1 to 1) and subjectivity (0 to 1)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity,
            'sentiment': 'Positive' if analysis.sentiment.polarity > 0 else 
                        'Negative' if analysis.sentiment.polarity < 0 else 'Neutral'
        }
    
    @staticmethod
    def analyze_review_sentiments(df):
        """Analyze sentiments for all reviews in the dataframe."""
        if 'review_text' not in df.columns:
            raise ValueError("DataFrame must contain 'review_text' column")
        
        # Apply sentiment analysis to each review
        sentiments = df['review_text'].apply(AdvancedAnalytics.analyze_sentiment)
        
        # Extract sentiment components
        df['sentiment_polarity'] = sentiments.apply(lambda x: x['polarity'])
        df['sentiment_subjectivity'] = sentiments.apply(lambda x: x['subjectivity'])
        df['sentiment_label'] = sentiments.apply(lambda x: x['sentiment'])
        
        return df

    @staticmethod
    def detect_sales_anomalies(df, contamination=0.1):
        """Detect anomalies in sales patterns using Isolation Forest."""
        if not all(col in df.columns for col in ['date', 'revenue']):
            raise ValueError("DataFrame must contain 'date' and 'revenue' columns")
        
        # Prepare data for anomaly detection
        daily_sales = df.groupby('date')['revenue'].sum().reset_index()
        daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        daily_sales['is_anomaly'] = iso_forest.fit_predict(daily_sales[['revenue', 'day_of_week']])
        daily_sales['is_anomaly'] = daily_sales['is_anomaly'].map({1: False, -1: True})
        
        return daily_sales

    @staticmethod
    def perform_cohort_analysis(df, time_window='M'):
        """Perform cohort analysis based on first purchase date."""
        if not all(col in df.columns for col in ['date', 'product_id', 'revenue']):
            raise ValueError("DataFrame must contain 'date', 'product_id', and 'revenue' columns")
        
        # Create cohort groups
        df['cohort_date'] = df.groupby('product_id')['date'].transform('min').dt.to_period(time_window)
        df['activity_period'] = df['date'].dt.to_period(time_window)
        df['periods_active'] = (df['activity_period'] - df['cohort_date']).apply(lambda x: x.n)
        
        # Calculate cohort metrics
        cohort_data = df.groupby(['cohort_date', 'periods_active']).agg({
            'product_id': 'nunique',
            'revenue': 'sum'
        }).reset_index()
        
        # Pivot table for cohort analysis
        cohort_sizes = cohort_data.pivot(
            index='cohort_date',
            columns='periods_active',
            values='product_id'
        )
        
        retention_rates = cohort_sizes.divide(cohort_sizes[0], axis=0)
        
        return {
            'cohort_sizes': cohort_sizes,
            'retention_rates': retention_rates
        }
