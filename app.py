import streamlit as st
import pandas as pd
from utils.data_processor import DataProcessor
from utils.visualizations import create_sales_charts, create_marketing_charts, create_review_charts
from utils.predictions import predict_trends, generate_recommendations
import io

# Page configuration
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = None
if 'marketing_data' not in st.session_state:
    st.session_state.marketing_data = None
if 'review_data' not in st.session_state:
    st.session_state.review_data = None

def main():
    st.title("📊 Analytics Dashboard")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Data Upload", "Sales Analytics", "Marketing Analytics", "Review Analytics", "Predictions"]
    )
    
    if page == "Data Upload":
        show_data_upload()
    elif page == "Sales Analytics":
        show_sales_analytics()
    elif page == "Marketing Analytics":
        show_marketing_analytics()
    elif page == "Review Analytics":
        show_review_analytics()
    elif page == "Predictions":
        show_predictions()

def show_data_upload():
    st.header("Data Upload")
    
    with st.expander("Data Format Requirements", expanded=True):
        st.markdown("""
        ### Required CSV Formats:
        
        **Sales Data:**
        - date: YYYY-MM-DD
        - product_id: string
        - quantity: integer
        - revenue: float
        
        **Marketing Data:**
        - date: YYYY-MM-DD
        - campaign_id: string
        - spend: float
        - impressions: integer
        - clicks: integer
        
        **Review Data:**
        - date: YYYY-MM-DD
        - product_id: string
        - rating: float (1-5)
        - review_text: string
        """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sales_file = st.file_uploader("Upload Sales Data (CSV)", type=['csv'])
        if sales_file:
            try:
                st.session_state.sales_data = DataProcessor.process_sales_data(sales_file)
                st.success("Sales data uploaded successfully!")
            except Exception as e:
                st.error(f"Error processing sales data: {str(e)}")
    
    with col2:
        marketing_file = st.file_uploader("Upload Marketing Data (CSV)", type=['csv'])
        if marketing_file:
            try:
                st.session_state.marketing_data = DataProcessor.process_marketing_data(marketing_file)
                st.success("Marketing data uploaded successfully!")
            except Exception as e:
                st.error(f"Error processing marketing data: {str(e)}")
    
    with col3:
        review_file = st.file_uploader("Upload Review Data (CSV)", type=['csv'])
        if review_file:
            try:
                st.session_state.review_data = DataProcessor.process_review_data(review_file)
                st.success("Review data uploaded successfully!")
            except Exception as e:
                st.error(f"Error processing review data: {str(e)}")

def show_sales_analytics():
    if st.session_state.sales_data is None:
        st.warning("Please upload sales data first!")
        return
    
    st.header("Sales Analytics")
    
    # Date range filter
    date_range = st.date_input(
        "Select Date Range",
        value=(st.session_state.sales_data['date'].min(),
               st.session_state.sales_data['date'].max()),
        key='sales_date_range'
    )
    
    # Create visualizations
    fig_sales = create_sales_charts(st.session_state.sales_data, date_range)
    st.plotly_chart(fig_sales, use_container_width=True)
    
    # Export functionality
    if st.button("Export Sales Data"):
        csv = st.session_state.sales_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="sales_data_export.csv",
            mime="text/csv"
        )

def show_marketing_analytics():
    if st.session_state.marketing_data is None:
        st.warning("Please upload marketing data first!")
        return
    
    st.header("Marketing Analytics")
    
    # Date range filter
    date_range = st.date_input(
        "Select Date Range",
        value=(st.session_state.marketing_data['date'].min(),
               st.session_state.marketing_data['date'].max()),
        key='marketing_date_range'
    )
    
    # Create visualizations
    fig_marketing = create_marketing_charts(st.session_state.marketing_data, date_range)
    st.plotly_chart(fig_marketing, use_container_width=True)
    
    # Export functionality
    if st.button("Export Marketing Data"):
        csv = st.session_state.marketing_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="marketing_data_export.csv",
            mime="text/csv"
        )

def show_review_analytics():
    if st.session_state.review_data is None:
        st.warning("Please upload review data first!")
        return
    
    st.header("Review Analytics")
    
    # Date range filter
    date_range = st.date_input(
        "Select Date Range",
        value=(st.session_state.review_data['date'].min(),
               st.session_state.review_data['date'].max()),
        key='review_date_range'
    )
    
    # Create visualizations
    fig_reviews = create_review_charts(st.session_state.review_data, date_range)
    st.plotly_chart(fig_reviews, use_container_width=True)
    
    # Export functionality
    if st.button("Export Review Data"):
        csv = st.session_state.review_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="review_data_export.csv",
            mime="text/csv"
        )

def show_predictions():
    if st.session_state.sales_data is None:
        st.warning("Please upload sales data first!")
        return
    
    st.header("Predictions and Recommendations")
    
    # Trend predictions
    st.subheader("Sales Trend Predictions")
    prediction_days = st.slider("Select prediction horizon (days)", 7, 90, 30)
    
    trends = predict_trends(st.session_state.sales_data, prediction_days)
    st.plotly_chart(trends, use_container_width=True)
    
    # Product recommendations
    st.subheader("Product Recommendations")
    recommendations = generate_recommendations(st.session_state.sales_data)
    st.write(recommendations)

if __name__ == "__main__":
    main()
