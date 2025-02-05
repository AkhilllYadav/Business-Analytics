import streamlit as st
import pandas as pd
from utils.data_processor import DataProcessor
from utils.visualizations import create_sales_charts, create_marketing_charts, create_review_charts
from utils.predictions import predict_trends, generate_recommendations
from utils.advanced_analytics import AdvancedAnalytics  # Added import
import io

# Page configuration and styling
st.set_page_config(page_title="Business Analytics Dashboard",
                   page_icon="📊",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Initialize session state
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = None
if 'marketing_data' not in st.session_state:
    st.session_state.marketing_data = None
if 'review_data' not in st.session_state:
    st.session_state.review_data = None


def main():
    # Sidebar with modern styling
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/dashboard-layout.png",
                 width=50)
        st.title("Navigation")

        # Theme toggle
        theme = st.select_slider(
            "Theme",
            options=['Light', 'Dark'],
            value='Light' if st.session_state.theme == 'light' else 'Dark')
        st.session_state.theme = theme.lower()

        # Navigation
        page = st.radio("", [
            "📥 Data Upload", "📈 Sales Analytics", "🎯 Marketing Analytics",
            "⭐ Review Analytics", "🔮 Predictions"
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

    st.header("🔮 Predictions and Recommendations")

    # Prediction controls
    st.markdown("""
        <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
            <h4 style='color: var(--text-primary);'>Prediction Settings</h4>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        prediction_days = st.slider(
            "Prediction Horizon (Days)",
            min_value=7,
            max_value=90,
            value=30,
            help="Choose the number of days to forecast into the future"
        )

    # Show predictions
    with st.spinner("Generating predictions..."):
        try:
            fig, predictions, ci_lower, ci_upper = predict_trends(st.session_state.sales_data, prediction_days)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

                # Show prediction metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Average Predicted Revenue",
                        f"${predictions.mean():,.2f}",
                        delta=f"{((predictions.mean() / st.session_state.sales_data['revenue'].mean() - 1) * 100):,.1f}%"
                    )
                with col2:
                    st.metric(
                        "Prediction Range",
                        f"${predictions.max():,.2f}",
                        f"${predictions.min():,.2f}"
                    )
                with col3:
                    confidence_range = f"${(ci_upper - ci_lower).mean():,.2f}"
                    st.metric("Average Confidence Range", confidence_range)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")


    # Show recommendations
    st.subheader("📋 Product Recommendations")
    with st.spinner("Generating recommendations..."):
        recommendations = generate_recommendations(st.session_state.sales_data)
        if recommendations is not None:
            # Display recommendations in a modern table
            st.markdown("""
                <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 1rem; margin: 1rem 0;'>
                    <h4 style='color: var(--text-primary);'>Top 5 Products by Performance</h4>
                </div>
            """, unsafe_allow_html=True)

            # Style the dataframe
            styled_recommendations = recommendations.style\
                .background_gradient(cmap='Blues', subset=['composite_score'])\
                .format({
                    'total_revenue': '${:,.2f}',
                    'avg_revenue': '${:,.2f}',
                    'composite_score': '{:.3f}'
                })

            st.dataframe(styled_recommendations, use_container_width=True)

            # Add explanation of the composite score
            st.info("""
                💡 The composite score is calculated based on:
                - Revenue performance (40%)
                - Sales quantity (40%)
                - Sales stability (20%)

                Lower scores indicate better overall performance.
            """)
        else:
            st.warning("No recommendations could be generated.")


if __name__ == "__main__":
    main()