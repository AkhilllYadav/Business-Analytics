# Business Analytics Dashboard

A comprehensive analytics dashboard built with Streamlit for visualizing and analyzing business metrics across sales, marketing, and customer reviews.

## Features

### 📥 Data Upload
- Support for CSV data uploads
- Automated data processing and validation
- Separate modules for sales, marketing, and review data

### 📈 Sales Analytics
- Daily revenue tracking
- Product performance analysis
- Sales quantity trends
- Revenue correlations

### 🎯 Marketing Analytics 
- Campaign performance metrics
- Click-through rate analysis
- Marketing spend optimization
- ROI tracking

### ⭐ Review Analytics
- Sentiment analysis
- Rating trends
- Product feedback analysis
- Customer satisfaction metrics

### 🔮 Advanced Predictions
- Sales forecasting
- Product recommendations
- Anomaly detection
- Performance scoring

### 🤖 AI-Powered Insights
- Executive summaries
- Customer behavior analysis
- Advanced sentiment analysis
- Marketing optimization suggestions

## Getting Started

1. Click the "Run" button to start the application
2. Use the login credentials to access the dashboard
3. Upload your CSV data files in the required format:

```
Sales Data:
- date: YYYY-MM-DD
- product_id: string
- quantity: integer
- revenue: float

Marketing Data:
- date: YYYY-MM-DD
- campaign_id: string
- spend: float
- impressions: integer
- clicks: integer

Review Data:
- date: YYYY-MM-DD
- product_id: string
- rating: float (1-5)
- review_text: string
```

## Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Analytics**: Scikit-learn, XGBoost
- **Visualization**: Plotly
- **AI/ML**: Google Gemini AI, TextBlob
- **Authentication**: Custom authentication system

## Requirements

All dependencies are automatically installed through pyproject.toml when you run the application.

## Security

The dashboard includes authentication protection and secure data handling. Credentials are required to access the analytics features.

## 🎥 Demo Video

Watch the demo video here: [Business Analytics Dashboard Video](https://player.vimeo.com/video/1053849253)


