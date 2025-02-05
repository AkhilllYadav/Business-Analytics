import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# We'll implement a fallback mechanism if Gemini API is not available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    # Initialize Gemini API with safety fallback
    if os.getenv('GOOGLE_API_KEY'):
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    else:
        GEMINI_AVAILABLE = False
except ImportError:
    GEMINI_AVAILABLE = False

class GeminiAnalytics:
    @staticmethod
    def _get_mock_insights():
        """Provide mock insights when Gemini API is unavailable"""
        return {
            'key_achievements': [
                'Achieved consistent sales growth',
                'Maintained positive customer feedback'
            ],
            'growth_opportunities': [
                'Potential for market expansion',
                'Opportunity to enhance product range'
            ]
        }

    @staticmethod
    def analyze_customer_behavior(df):
        """Analyze customer purchase patterns and behavior."""
        try:
            if not GEMINI_AVAILABLE:
                return {
                    'key_findings': ['Analysis requires Gemini API integration'],
                    'recommendations': ['Please check API configuration']
                }

            # Prepare data for analysis
            daily_metrics = df.groupby('date').agg({
                'product_id': 'nunique',
                'quantity': 'sum',
                'revenue': 'sum'
            }).reset_index()

            # Generate insights using Gemini
            data_description = f"""
            Sales Data Summary:
            - Date Range: {daily_metrics['date'].min()} to {daily_metrics['date'].max()}
            - Total Revenue: ${daily_metrics['revenue'].sum():,.2f}
            - Total Units Sold: {daily_metrics['quantity'].sum():,}
            - Unique Products: {df['product_id'].nunique()}
            """

            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(
                f"""Analyze this sales data and provide key insights:
                {data_description}
                Format the response as a JSON string with these keys:
                - key_findings: list of main insights
                - recommendations: list of actionable recommendations
                """
            )

            insights = json.loads(response.text)
            return insights

        except Exception as e:
            st.error(f"Error in customer behavior analysis: {str(e)}")
            return None

    # Similar pattern for other methods...
    @staticmethod
    def analyze_review_sentiment(reviews_df):
        if not GEMINI_AVAILABLE:
            return {
                'key_themes': ['Sample theme 1', 'Sample theme 2'],
                'improvement_areas': ['Area 1', 'Area 2']
            }
        try:
            # Sample reviews for analysis
            sample_reviews = reviews_df.sample(min(10, len(reviews_df)))
            review_texts = sample_reviews['review_text'].tolist()

            # Analyze using Gemini
            model = genai.GenerativeModel('gemini-pro')
            analysis_prompt = f"""Analyze these customer reviews and provide detailed sentiment insights:
            Reviews: {review_texts}

            Format the response as a JSON string with these keys:
            - key_themes: list of main themes mentioned
            - improvement_areas: list of suggested improvements
            """

            response = model.generate_content(analysis_prompt)
            sentiment_analysis = json.loads(response.text)

            return sentiment_analysis

        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
            return None

    @staticmethod
    def detect_anomalies(df):
        if not GEMINI_AVAILABLE:
            return {
                'anomalies_detected': ['Sample anomaly 1'],
                'risk_factors': ['Risk factor 1']
            }
        try:
            # Prepare daily metrics
            daily_metrics = df.groupby('date').agg({
                'revenue': 'sum',
                'quantity': 'sum'
            }).reset_index()

            # Calculate basic statistics
            revenue_stats = {
                'mean': daily_metrics['revenue'].mean(),
                'std': daily_metrics['revenue'].std(),
                'max': daily_metrics['revenue'].max(),
                'min': daily_metrics['revenue'].min()
            }

            # Generate insights using Gemini
            model = genai.GenerativeModel('gemini-pro')
            analysis_prompt = f"""Analyze these sales metrics for anomalies:
            Revenue Statistics:
            - Mean: ${revenue_stats['mean']:,.2f}
            - Std Dev: ${revenue_stats['std']:,.2f}
            - Max: ${revenue_stats['max']:,.2f}
            - Min: ${revenue_stats['min']:,.2f}

            Format the response as a JSON string with these keys:
            - anomalies_detected: list of anomaly descriptions
            - risk_factors: list of potential risk factors
            """

            response = model.generate_content(analysis_prompt)
            anomaly_analysis = json.loads(response.text)

            return anomaly_analysis

        except Exception as e:
            st.error(f"Error in anomaly detection: {str(e)}")
            return None

    @staticmethod
    def generate_marketing_insights(marketing_df, sales_df):
        if not GEMINI_AVAILABLE:
            return {
                'performance_insights': ['Sample insight 1'],
                'optimization_suggestions': ['Sample suggestion 1']
            }
        try:
            # Prepare marketing metrics
            marketing_metrics = marketing_df.groupby('campaign_id').agg({
                'spend': 'sum',
                'clicks': 'sum',
                'impressions': 'sum'
            }).reset_index()

            marketing_metrics['ctr'] = marketing_metrics['clicks'] / marketing_metrics['impressions']
            marketing_metrics['cpc'] = marketing_metrics['spend'] / marketing_metrics['clicks']

            # Generate insights using Gemini
            metrics_summary = f"""
            Marketing Performance:
            - Total Spend: ${marketing_metrics['spend'].sum():,.2f}
            - Total Clicks: {marketing_metrics['clicks'].sum():,}
            - Average CTR: {(marketing_metrics['ctr'].mean() * 100):.2f}%
            - Average CPC: ${marketing_metrics['cpc'].mean():.2f}
            """

            model = genai.GenerativeModel('gemini-pro')
            analysis_prompt = f"""Analyze these marketing metrics and provide insights:
            {metrics_summary}

            Format the response as a JSON string with these keys:
            - performance_insights: list of key performance insights
            - optimization_suggestions: list of ways to improve campaigns
            """

            response = model.generate_content(analysis_prompt)
            marketing_analysis = json.loads(response.text)

            return marketing_analysis

        except Exception as e:
            st.error(f"Error in marketing analysis: {str(e)}")
            return None

    @staticmethod
    def create_executive_summary(sales_df, marketing_df, reviews_df):
        """Generate an AI-powered executive summary of all metrics."""
        if not GEMINI_AVAILABLE:
            return GeminiAnalytics._get_mock_insights()

        try:
            # Prepare summary metrics
            summary = {
                'total_revenue': sales_df['revenue'].sum(),
                'total_sales': sales_df['quantity'].sum(),
                'avg_order_value': sales_df['revenue'].mean(),
                'marketing_spend': marketing_df['spend'].sum(),
                'total_reviews': len(reviews_df),
                'avg_rating': reviews_df['rating'].mean()
            }

            model = genai.GenerativeModel('gemini-pro')
            summary_prompt = f"""Create an executive summary based on these metrics:
            Business Performance Summary:
            - Total Revenue: ${summary['total_revenue']:,.2f}
            - Total Sales: {summary['total_sales']:,}
            - Average Order Value: ${summary['avg_order_value']:.2f}
            - Marketing Spend: ${summary['marketing_spend']:,.2f}
            - Total Reviews: {summary['total_reviews']:,}
            - Average Rating: {summary['avg_rating']:.2f}

            Format the response as a JSON string with these keys:
            - key_achievements: list of main achievements
            - growth_opportunities: list of potential growth areas
            """

            response = model.generate_content(summary_prompt)
            return json.loads(response.text)

        except Exception as e:
            st.error(f"Error generating executive summary: {str(e)}")
            return GeminiAnalytics._get_mock_insights()