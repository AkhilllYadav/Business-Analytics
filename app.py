import streamlit as st
import pandas as pd
from utils.data_processor import DataProcessor
from utils.visualizations import create_sales_charts, create_marketing_charts, create_review_charts
from utils.predictions import AdvancedPredictions
from utils.advanced_analytics import AdvancedAnalytics
from utils.gemini_analytics import GeminiAnalytics
from utils.auth import check_password
import io
import numpy as np

# Page configuration and styling
st.set_page_config(page_title="Business Analytics Dashboard",
                   page_icon="📊",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Initialize session state for data
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = None
if 'marketing_data' not in st.session_state:
    st.session_state.marketing_data = None
if 'review_data' not in st.session_state:
    st.session_state.review_data = None

def main():
    # Check authentication before showing any content
    if not check_password():
        return

    # Rest of your main function remains unchanged
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/dashboard-layout.png",
                 width=50)
        st.title("Navigation")

        # Add logout button in sidebar
        if st.button("Logout"):
            st.session_state["password_correct"] = False
            st.rerun()

        # Theme toggle
        theme = st.select_slider(
            "Theme",
            options=['Light', 'Dark'],
            value='Light' if st.session_state.theme == 'light' else 'Dark')
        st.session_state.theme = theme.lower()

        # Navigation
        page = st.radio("", [
            "📥 Data Upload", "📈 Sales Analytics", "🎯 Marketing Analytics",
            "⭐ Review Analytics", "🔮 Predictions", "🤖 AI Insights"
        ])

    # Main content area with modern header
    st.markdown(f"""
        <div style='text-align: center; padding: 1rem;'>
            <h1>Business Analytics Dashboard</h1>
            <p style='color: var(--text-secondary);'>Transform your data into actionable insights</p>
        </div>
    """,
                unsafe_allow_html=True)

    if "Data Upload" in page:
        show_data_upload()
    elif "Sales Analytics" in page:
        show_sales_analytics()
    elif "Marketing Analytics" in page:
        show_marketing_analytics()
    elif "Review Analytics" in page:
        show_review_analytics()
    elif "Predictions" in page:
        show_predictions()
    elif "AI Insights" in page:
        show_ai_insights()


def show_data_upload():
    st.header("📥 Data Upload")

    with st.expander("ℹ️ Data Format Requirements", expanded=True):
        st.markdown("""
        ### Required CSV Formats

        **📊 Sales Data:**
        - `date`: YYYY-MM-DD
        - `product_id`: string
        - `quantity`: integer
        - `revenue`: float

        **🎯 Marketing Data:**
        - `date`: YYYY-MM-DD
        - `campaign_id`: string
        - `spend`: float
        - `impressions`: integer
        - `clicks`: integer

        **⭐ Review Data:**
        - `date`: YYYY-MM-DD
        - `product_id`: string
        - `rating`: float (1-5)
        - `review_text`: string
        """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div style='background-color: var(--card-bg); padding: 1.5rem; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3 style='text-align: center; color: var(--text-primary);'>Sales Data</h3>
            </div>
        """,
                    unsafe_allow_html=True)
        sales_file = st.file_uploader("", type=['csv'], key='sales_upload')
        if sales_file:
            try:
                st.session_state.sales_data = DataProcessor.process_sales_data(
                    sales_file)
                st.success("✅ Sales data uploaded successfully!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    with col2:
        st.markdown("""
            <div style='background-color: var(--card-bg); padding: 1.5rem; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3 style='text-align: center; color: var(--text-primary);'>Marketing Data</h3>
            </div>
        """,
                    unsafe_allow_html=True)
        marketing_file = st.file_uploader("",
                                          type=['csv'],
                                          key='marketing_upload')
        if marketing_file:
            try:
                st.session_state.marketing_data = DataProcessor.process_marketing_data(
                    marketing_file)
                st.success("✅ Marketing data uploaded successfully!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    with col3:
        st.markdown("""
            <div style='background-color: var(--card-bg); padding: 1.5rem; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3 style='text-align: center; color: var(--text-primary);'>Review Data</h3>
            </div>
        """,
                    unsafe_allow_html=True)
        review_file = st.file_uploader("", type=['csv'], key='review_upload')
        if review_file:
            try:
                st.session_state.review_data = DataProcessor.process_review_data(
                    review_file)
                st.success("✅ Review data uploaded successfully!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def show_sales_analytics():
    if st.session_state.sales_data is None:
        st.warning("⚠️ Please upload sales data first!")
        return

    st.header("📈 Sales Analytics")

    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    total_revenue = st.session_state.sales_data['revenue'].sum()
    total_sales = st.session_state.sales_data['quantity'].sum()
    avg_order_value = total_revenue / len(st.session_state.sales_data) if len(
        st.session_state.sales_data) > 0 else 0
    unique_products = st.session_state.sales_data['product_id'].nunique()

    with col1:
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    with col2:
        st.metric("Total Sales", f"{total_sales:,}")
    with col3:
        st.metric("Avg Order Value", f"${avg_order_value:,.2f}")
    with col4:
        st.metric("Unique Products", unique_products)

    # Date range filter with modern styling
    st.markdown("""
        <div style='background-color: var(--card-bg); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
            <h4 style='color: var(--text-primary);'>Select Date Range</h4>
        </div>
    """,
                unsafe_allow_html=True)
    date_range = st.date_input(
        "",
        value=(st.session_state.sales_data['date'].min(),
               st.session_state.sales_data['date'].max()),
        key='sales_date_range')

    # Create visualizations
    fig_sales = create_sales_charts(st.session_state.sales_data, date_range)
    st.plotly_chart(fig_sales, use_container_width=True)


def show_marketing_analytics():
    if st.session_state.marketing_data is None:
        st.warning("⚠️ Please upload marketing data first!")
        return

    st.header("🎯 Marketing Analytics")

    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    total_spend = st.session_state.marketing_data['spend'].sum()
    total_impressions = st.session_state.marketing_data['impressions'].sum()
    total_clicks = st.session_state.marketing_data['clicks'].sum()
    avg_ctr = (total_clicks / total_impressions *
               100) if total_impressions > 0 else 0

    with col1:
        st.metric("Total Spend", f"${total_spend:,.2f}")
    with col2:
        st.metric("Total Impressions", f"{total_impressions:,}")
    with col3:
        st.metric("Total Clicks", f"{total_clicks:,}")
    with col4:
        st.metric("Average CTR", f"{avg_ctr:.2f}%")

    # Date range filter
    st.markdown("""
        <div style='background-color: var(--card-bg); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
            <h4 style='color: var(--text-primary);'>Select Date Range</h4>
        </div>
    """,
                unsafe_allow_html=True)
    date_range = st.date_input(
        "",
        value=(st.session_state.marketing_data['date'].min(),
               st.session_state.marketing_data['date'].max()),
        key='marketing_date_range')

    # Create visualizations
    fig_marketing = create_marketing_charts(st.session_state.marketing_data,
                                            date_range)
    st.plotly_chart(fig_marketing, use_container_width=True)


def show_review_analytics():
    if st.session_state.review_data is None:
        st.warning("⚠️ Please upload review data first!")
        return

    st.header("⭐ Review Analytics")

    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)

    # Process sentiment analysis
    with st.spinner("Analyzing sentiments..."):
        review_data_with_sentiment = AdvancedAnalytics.analyze_review_sentiments(
            st.session_state.review_data.copy())

    avg_rating = review_data_with_sentiment['rating'].mean()
    total_reviews = len(review_data_with_sentiment)
    positive_sentiments = len(review_data_with_sentiment[
        review_data_with_sentiment['sentiment_label'] == 'Positive'])
    avg_sentiment = review_data_with_sentiment['sentiment_polarity'].mean()

    with col1:
        st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
    with col2:
        st.metric("Total Reviews", f"{total_reviews:,}")
    with col3:
        st.metric("Positive Sentiments", f"{positive_sentiments:,}")
    with col4:
        st.metric("Avg Sentiment Score", f"{avg_sentiment:.2f}")

    # Sentiment Analysis Section
    st.subheader("📊 Sentiment Analysis")

    # Sentiment distribution
    sentiment_dist = pd.DataFrame({
        'Sentiment':
        review_data_with_sentiment['sentiment_label'].value_counts(),
        'Percentage':
        review_data_with_sentiment['sentiment_label'].value_counts(
            normalize=True) * 100
    })

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
                <h4 style='color: var(--text-primary);'>Sentiment Distribution</h4>
            </div>
        """,
                    unsafe_allow_html=True)
        st.dataframe(sentiment_dist, use_container_width=True)

    with col2:
        st.markdown("""
            <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
                <h4 style='color: var(--text-primary);'>Sentiment vs Rating</h4>
            </div>
        """,
                    unsafe_allow_html=True)
        sentiment_vs_rating = review_data_with_sentiment.groupby(
            'rating')['sentiment_polarity'].mean()
        st.line_chart(sentiment_vs_rating)

    # Show sample reviews with sentiment
    st.subheader("📝 Sample Reviews with Sentiment Analysis")
    sample_reviews = review_data_with_sentiment.sample(
        min(5, len(review_data_with_sentiment)))
    for _, review in sample_reviews.iterrows():
        sentiment_color = (
            "🟢" if review['sentiment_label'] == 'Positive' else
            "🔴" if review['sentiment_label'] == 'Negative' else "⚪")
        st.markdown(f"""
            <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                <p style='color: var(--text-primary); margin: 0;'>
                    {sentiment_color} Rating: {review['rating']} | Sentiment Score: {review['sentiment_polarity']:.2f}
                </p>
                <p style='color: var(--text-secondary); margin: 0.5rem 0;'>"{review['review_text']}"</p>
            </div>
        """,
                    unsafe_allow_html=True)

    # Date range filter
    st.markdown("""
        <div style='background-color: var(--card-bg); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
            <h4 style='color: var(--text-primary);'>Select Date Range</h4>
        </div>
    """,
                unsafe_allow_html=True)
    date_range = st.date_input(
        "",
        value=(st.session_state.review_data['date'].min(),
               st.session_state.review_data['date'].max()),
        key='review_date_range')

    # Create visualizations
    fig_reviews = create_review_charts(st.session_state.review_data,
                                       date_range)
    st.plotly_chart(fig_reviews, use_container_width=True)


def show_predictions():
    if st.session_state.sales_data is None:
        st.warning("⚠️ Please upload sales data first!")
        return

    st.header("🔮 Advanced Analytics & Recommendations")

    # Prediction controls
    st.markdown("""
        <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
            <h4 style='color: var(--text-primary);'>Product Performance Analysis</h4>
        </div>
    """, unsafe_allow_html=True)

    # Show recommendations
    st.subheader("📊 Top Product Recommendations")
    with st.spinner("Analyzing product performance..."):
        try:
            recommendations = AdvancedPredictions.generate_recommendations(
                st.session_state.sales_data)

            if recommendations is not None:
                # Create visualization
                fig = AdvancedPredictions.create_recommendation_visualizations(recommendations)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

                # Display detailed recommendations
                st.markdown("""
                    <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
                        <h4 style='color: var(--text-primary);'>Detailed Product Analysis</h4>
                    </div>
                """, unsafe_allow_html=True)

                # Style the dataframe
                styled_recommendations = recommendations.style \
                    .background_gradient(cmap='Blues', subset=['composite_score']) \
                    .background_gradient(cmap='RdYlGn', subset=['revenue_growth', 'quantity_growth']) \
                    .format({
                        'total_revenue': '${:,.2f}',
                        'avg_transaction_value': '${:,.2f}',
                        'revenue_growth': '{:.1%}',
                        'quantity_growth': '{:.1%}',
                        'revenue_stability': '{:.2f}',
                        'composite_score': '{:.3f}'
                    })

                st.dataframe(styled_recommendations, use_container_width=True)

                # Add explanation
                st.info("""
                    💡 **Understanding the Metrics:**
                    - **Total Revenue**: Total sales revenue generated by the product
                    - **Avg Transaction Value**: Average revenue per transaction
                    - **Revenue Growth**: Month-over-month revenue growth rate
                    - **Quantity Growth**: Month-over-month sales volume growth rate
                    - **Revenue Stability**: Score between 0-1 indicating revenue consistency
                    - **Composite Score**: Overall performance score (lower is better)

                    The composite score weighs multiple factors:
                    - Revenue Performance (30%)
                    - Sales Volume (25%)
                    - Revenue Stability (20%)
                    - Volume Stability (15%)
                    - Transaction Value (10%)
                """)

                # Product-specific insights
                st.subheader("🔍 Product Insights")
                for _, product in recommendations.iterrows():
                    st.markdown(f"""
                        <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                            <h5 style='color: var(--text-primary);'>Product {product['product_id']}</h5>
                            <p style='color: var(--text-secondary);'>{product['performance_insight']}</p>
                        </div>
                    """, unsafe_allow_html=True)

            else:
                st.warning("Unable to generate recommendations. Please check your data.")

        except Exception as e:
            st.error(f"An error occurred while analyzing products: {str(e)}")


def show_ai_insights():
    if any(data is None for data in [st.session_state.sales_data,
                                    st.session_state.marketing_data,
                                    st.session_state.review_data]):
        st.warning("⚠️ Please upload all data (sales, marketing, and reviews) first!")
        return

    st.header("🤖 AI-Powered Analytics Insights")

    # Executive Summary
    st.subheader("📊 Executive Summary")
    with st.spinner("Generating executive summary..."):
        summary = GeminiAnalytics.create_executive_summary(
            st.session_state.sales_data,
            st.session_state.marketing_data,
            st.session_state.review_data
        )

        if summary:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                    <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
                        <h4 style='color: var(--text-primary);'>Key Achievements</h4>
                    </div>
                """, unsafe_allow_html=True)
                for achievement in summary['key_achievements']:
                    st.success(achievement)

            with col2:
                st.markdown("""
                    <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
                        <h4 style='color: var(--text-primary);'>Growth Opportunities</h4>
                    </div>
                """, unsafe_allow_html=True)
                for opportunity in summary['growth_opportunities']:
                    st.info(opportunity)

    # Customer Behavior Analysis
    st.subheader("👥 Customer Behavior Analysis")
    with st.spinner("Analyzing customer behavior..."):
        behavior_insights = GeminiAnalytics.analyze_customer_behavior(
            st.session_state.sales_data
        )

        if behavior_insights:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                    <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
                        <h4 style='color: var(--text-primary);'>Key Findings</h4>
                    </div>
                """, unsafe_allow_html=True)
                for finding in behavior_insights['key_findings']:
                    st.write(f"• {finding}")

            with col2:
                st.markdown("""
                    <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
                        <h4 style='color: var(--text-primary);'>Recommendations</h4>
                    </div>
                """, unsafe_allow_html=True)
                for rec in behavior_insights['recommendations']:
                    st.write(f"• {rec}")

    # Advanced Sentiment Analysis
    st.subheader("💭 Advanced Sentiment Analysis")
    with st.spinner("Performing sentiment analysis..."):
        sentiment_insights = GeminiAnalytics.analyze_review_sentiment(
            st.session_state.review_data
        )

        if sentiment_insights:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                    <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
                        <h4 style='color: var(--text-primary);'>Key Themes</h4>
                    </div>
                """, unsafe_allow_html=True)
                for theme in sentiment_insights['key_themes']:
                    st.write(f"• {theme}")

            with col2:
                st.markdown("""
                    <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
                        <h4 style='color: var(--text-primary);'>Areas for Improvement</h4>
                    </div>
                """, unsafe_allow_html=True)
                for area in sentiment_insights['improvement_areas']:
                    st.write(f"• {area}")

    # Anomaly Detection
    st.subheader("⚠️ Anomaly Detection")
    with st.spinner("Detecting anomalies..."):
        anomalies = GeminiAnalytics.detect_anomalies(
            st.session_state.sales_data
        )

        if anomalies:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                    <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
                        <h4 style='color: var(--text-primary);'>Detected Anomalies</h4>
                    </div>
                """, unsafe_allow_html=True)
                for anomaly in anomalies['anomalies_detected']:
                    st.warning(anomaly)

            with col2:
                st.markdown("""
                    <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
                        <h4 style='color: var(--text-primary);'>Risk Factors</h4>
                    </div>
                """, unsafe_allow_html=True)
                for risk in anomalies['risk_factors']:
                    st.error(risk)

    # Marketing Performance Analysis
    st.subheader("📢 AI-Driven Marketing Insights")
    with st.spinner("Analyzing marketing performance..."):
        marketing_insights = GeminiAnalytics.generate_marketing_insights(
            st.session_state.marketing_data,
            st.session_state.sales_data
        )

        if marketing_insights:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                    <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
                        <h4 style='color: var(--text-primary);'>Performance Insights</h4>
                    </div>
                """, unsafe_allow_html=True)
                for insight in marketing_insights['performance_insights']:
                    st.write(f"• {insight}")

            with col2:
                st.markdown("""
                    <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
                        <h4 style='color: var(--text-primary);'>Optimization Suggestions</h4>
                    </div>
                """, unsafe_allow_html=True)
                for suggestion in marketing_insights['optimization_suggestions']:
                    st.write(f"• {suggestion}")

if __name__ == "__main__":
    main()