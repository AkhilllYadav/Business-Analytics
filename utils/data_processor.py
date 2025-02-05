import pandas as pd
import numpy as np

class DataProcessor:
    @staticmethod
    def process_sales_data(file):
        """Process and validate sales data CSV."""
        try:
            df = pd.read_csv(file)
            required_columns = ['date', 'product_id', 'quantity', 'revenue']
            
            # Check for required columns
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required columns in sales data")
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            
            # Validate numeric columns
            df['quantity'] = pd.to_numeric(df['quantity'])
            df['revenue'] = pd.to_numeric(df['revenue'])
            
            # Basic data cleaning
            df = df.dropna()
            df = df[df['quantity'] >= 0]
            df = df[df['revenue'] >= 0]
            
            return df
            
        except Exception as e:
            raise Exception(f"Error processing sales data: {str(e)}")
    
    @staticmethod
    def process_marketing_data(file):
        """Process and validate marketing data CSV."""
        try:
            df = pd.read_csv(file)
            required_columns = ['date', 'campaign_id', 'spend', 'impressions', 'clicks']
            
            # Check for required columns
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required columns in marketing data")
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            
            # Validate numeric columns
            df['spend'] = pd.to_numeric(df['spend'])
            df['impressions'] = pd.to_numeric(df['impressions'])
            df['clicks'] = pd.to_numeric(df['clicks'])
            
            # Basic data cleaning
            df = df.dropna()
            df = df[df['spend'] >= 0]
            df = df[df['impressions'] >= 0]
            df = df[df['clicks'] >= 0]
            
            # Calculate CTR
            df['ctr'] = (df['clicks'] / df['impressions']).fillna(0)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error processing marketing data: {str(e)}")
    
    @staticmethod
    def process_review_data(file):
        """Process and validate review data CSV."""
        try:
            df = pd.read_csv(file)
            required_columns = ['date', 'product_id', 'rating', 'review_text']
            
            # Check for required columns
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required columns in review data")
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            
            # Validate rating column
            df['rating'] = pd.to_numeric(df['rating'])
            df = df[df['rating'].between(1, 5)]
            
            # Basic data cleaning
            df = df.dropna(subset=['date', 'product_id', 'rating'])
            
            return df
            
        except Exception as e:
            raise Exception(f"Error processing review data: {str(e)}")
