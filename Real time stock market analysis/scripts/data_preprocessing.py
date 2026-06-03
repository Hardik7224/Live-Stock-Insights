import os
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from scripts.indicators import enrich_with_indicators

# Configure logging
logger = logging.getLogger("DataPreprocessing")


class StockDataPreprocessor:
    """
    A professional class to clean, preprocess, and engineer financial features
    on raw historical and real-time stock datasets.
    """

    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.processed_data_dir, exist_ok=True)
        logger.info(f"Initialized StockDataPreprocessor. Output processed directory: {self.processed_data_dir}")

    def detect_outliers_iqr(self, df: pd.DataFrame, column: str, multiplier: float = 3.0) -> Tuple[pd.DataFrame, int]:
        """
        Identifies outliers using the Interquartile Range (IQR) method.
        Note: We flag outliers but do not drop them, as extreme returns and volumes 
        represent real financial market events (earnings releases, market shocks).
        
        Args:
            df (pd.DataFrame): The DataFrame.
            column (str): The column to check.
            multiplier (float): The IQR factor. Usually 1.5 for normal outliers, 3.0 for extreme.

        Returns:
            Tuple[pd.DataFrame, int]: DataFrame with a new boolean outlier column, and count of outliers.
        """
        if column not in df.columns or df[column].isnull().all():
            return df, 0

        q25 = df[column].quantile(0.25)
        q75 = df[column].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - (multiplier * iqr)
        upper_bound = q75 + (multiplier * iqr)

        outlier_col = f"{column}_Outlier"
        df[outlier_col] = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_count = int(df[outlier_col].sum())

        logger.debug(f"Outlier detection on {column}: found {outlier_count} outliers (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")
        return df, outlier_count

    def preprocess_historical_file(self, file_path: str) -> Optional[str]:
        """
        Cleans and enriches a single raw historical stock CSV file.
        
        Steps:
        1. Loads the CSV data
        2. Deduplicates rows based on date
        3. Parses Date as Index
        4. Fills missing price bars using forward-fill (FFill) and backward-fill (BFill)
        5. Engineers Daily Returns & Volatility
        6. Integrates Technical Indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
        7. Detects and flags Volatility and Volume outliers
        8. Saves processed CSV
        """
        try:
            logger.info(f"Preprocessing raw file: {file_path}")
            df = pd.read_csv(file_path)
            
            if df.empty:
                logger.warning(f"File {file_path} is empty. Skipping.")
                return None

            ticker = df['Ticker'].iloc[0]

            # 1. Date parsing and sorting
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)

            # 2. Deduplicate
            initial_rows = len(df)
            df = df.drop_duplicates(subset=['Date', 'Ticker']).reset_index(drop=True)
            dup_removed = initial_rows - len(df)
            if dup_removed > 0:
                logger.info(f"Removed {dup_removed} duplicate rows for ticker {ticker}.")

            # 3. Handle missing values
            # Forward fill prices (carry forward last known trade), then backfill if first few rows are empty
            price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if 'Adj Close' in df.columns:
                price_cols.append('Adj Close')
            
            nulls_before = df[price_cols].isnull().sum().sum()
            if nulls_before > 0:
                df[price_cols] = df[price_cols].ffill().bfill()
                logger.info(f"Handled {nulls_before} null values using forward/backward fill.")

            # 4. Feature Engineering
            # Daily Return & Percentage Change
            df['Daily_Return'] = df['Close'].pct_change()
            df['Pct_Change'] = df['Daily_Return'] * 100
            
            # Volatility (20-day rolling standard deviation of daily returns)
            df['Volatility_20d'] = df['Daily_Return'].rolling(window=20).std()

            # Rolling stats on Close & Volume
            df['Close_Rolling_Mean_20'] = df['Close'].rolling(window=20).mean()
            df['Close_Rolling_Std_20'] = df['Close'].rolling(window=20).std()
            df['Volume_Rolling_Mean_20'] = df['Volume'].rolling(window=20).mean()

            # Fill first row daily return NaN
            df['Daily_Return'] = df['Daily_Return'].fillna(0)
            df['Pct_Change'] = df['Pct_Change'].fillna(0)

            # 5. Enrich with Technical Indicators
            df = enrich_with_indicators(df, column="Close")

            # 6. Flag outliers (returns and volume)
            df, return_outliers = self.detect_outliers_iqr(df, "Daily_Return", multiplier=3.0)
            df, volume_outliers = self.detect_outliers_iqr(df, "Volume", multiplier=3.0)
            logger.info(f"{ticker} Outlier Summary: Daily_Return outliers = {return_outliers}, Volume outliers = {volume_outliers}")

            # 7. Save output
            output_file_name = f"{ticker.replace('.', '_')}_historical_processed.csv"
            output_path = os.path.join(self.processed_data_dir, output_file_name)
            df.to_csv(output_path, index=False)
            logger.info(f"Successfully preprocessed {ticker} historical data ({len(df)} rows) -> {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to preprocess file {file_path}: {str(e)}")
            return None

    def preprocess_realtime_file(self, file_path: str) -> Optional[str]:
        """
        Cleans and enriches a single raw real-time (intraday) stock CSV file.
        Uses similar preprocessing steps modified for datetime-grained data.
        """
        try:
            logger.info(f"Preprocessing raw real-time file: {file_path}")
            df = pd.read_csv(file_path)

            if df.empty:
                logger.warning(f"File {file_path} is empty. Skipping.")
                return None

            ticker = df['Ticker'].iloc[0]

            # Parse datetime
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df = df.sort_values('Datetime').reset_index(drop=True)

            # Deduplicate
            initial_rows = len(df)
            df = df.drop_duplicates(subset=['Datetime', 'Ticker']).reset_index(drop=True)
            dup_removed = initial_rows - len(df)
            if dup_removed > 0:
                logger.info(f"Removed {dup_removed} duplicate rows for ticker {ticker}.")

            # Clean and fill nulls
            price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[price_cols] = df[price_cols].ffill().bfill()

            # Engineers
            df['Daily_Return'] = df['Close'].pct_change().fillna(0)
            df['Pct_Change'] = df['Daily_Return'] * 100
            
            # Since real-time intraday data is high-frequency, rolling volatility is measured on shorter windows
            df['Volatility_5m'] = df['Daily_Return'].rolling(window=5).std().fillna(0)

            # Enrich with Technical Indicators (shortened indicators or default)
            df = enrich_with_indicators(df, column="Close")

            # Outliers
            df, return_outliers = self.detect_outliers_iqr(df, "Daily_Return", multiplier=3.0)

            # Save
            output_file_name = f"{ticker.replace('.', '_')}_realtime_processed.csv"
            output_path = os.path.join(self.processed_data_dir, output_file_name)
            df.to_csv(output_path, index=False)
            logger.info(f"Successfully preprocessed {ticker} real-time data ({len(df)} rows) -> {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to preprocess real-time file {file_path}: {str(e)}")
            return None


if __name__ == "__main__":
    # Test execution
    preprocessor = StockDataPreprocessor(raw_data_dir="data/raw", processed_data_dir="data/processed")
    
    # We will test preprocessing files once data collection has run
    logger.info("Ready to preprocess. Import and run via main ETL script.")
