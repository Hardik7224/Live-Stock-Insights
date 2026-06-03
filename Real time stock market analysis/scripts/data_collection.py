import os
import logging
from typing import List, Optional
import pandas as pd
import yfinance as yf

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("DataCollection")


class StockDataCollector:
    """
    A professional class to fetch stock market data using the yfinance API.
    Supports historical and real-time extraction for multiple tickers.
    """

    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = raw_data_dir
        os.makedirs(self.raw_data_dir, exist_ok=True)
        logger.info(f"Initialized StockDataCollector. Raw data directory set to: {self.raw_data_dir}")

    def fetch_historical_data(
        self, 
        tickers: List[str], 
        period: str = "5y", 
        interval: str = "1d"
    ) -> List[str]:
        """
        Fetches historical daily stock data from yfinance and saves it to CSV.

        Args:
            tickers (List[str]): List of stock symbols (e.g., ['AAPL', 'RELIANCE.NS'])
            period (str): Date range to fetch. Default '5y'.
            interval (str): Frequency of stock bars. Default '1d'.

        Returns:
            List[str]: List of paths to the saved raw historical CSV files.
        """
        saved_paths = []
        logger.info(f"Starting historical data collection for tickers: {tickers} over period: {period}")

        for ticker in tickers:
            try:
                logger.info(f"Fetching historical data for {ticker}...")
                # Fetch data
                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(period=period, interval=interval)

                if df.empty:
                    logger.warning(f"No historical data returned for ticker {ticker}. Skipping.")
                    continue

                # Add metadata columns
                df = df.reset_index()
                df['Ticker'] = ticker
                
                # Check for standard columns and rename appropriately if needed
                # yfinance returns 'Date' (or 'Datetime'), 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'
                # Ensure date column is formatted as standard Date string
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date']).dt.date
                elif 'Datetime' in df.columns:
                    df['Date'] = pd.to_datetime(df['Datetime']).dt.date
                    df = df.drop(columns=['Datetime'])

                # Save file
                file_name = f"{ticker.replace('.', '_')}_historical.csv"
                file_path = os.path.join(self.raw_data_dir, file_name)
                df.to_csv(file_path, index=False)
                
                saved_paths.append(file_path)
                logger.info(f"Successfully saved {ticker} historical data ({len(df)} rows) to {file_path}")

            except Exception as e:
                logger.error(f"Error fetching historical data for ticker {ticker}: {str(e)}")

        return saved_paths

    def fetch_realtime_data(
        self, 
        tickers: List[str], 
        period: str = "1d", 
        interval: str = "1m"
    ) -> List[str]:
        """
        Fetches the latest intraday stock data (near real-time) from yfinance and saves it to CSV.

        Args:
            tickers (List[str]): List of stock symbols.
            period (str): Date range for intraday. Default '1d'.
            interval (str): Small interval for live charting. Default '1m'.

        Returns:
            List[str]: List of paths to the saved raw real-time CSV files.
        """
        saved_paths = []
        logger.info(f"Starting real-time data collection for tickers: {tickers} with interval: {interval}")

        for ticker in tickers:
            try:
                logger.info(f"Fetching real-time data for {ticker}...")
                ticker_obj = yf.Ticker(ticker)
                
                # Intraday data
                df = ticker_obj.history(period=period, interval=interval)

                if df.empty:
                    logger.warning(f"No real-time data returned for ticker {ticker}. Skipping.")
                    continue

                df = df.reset_index()
                df['Ticker'] = ticker

                # Format Date/Datetime column
                # Intraday data will have Datetime (timezone-aware)
                if 'Datetime' in df.columns:
                    df['Datetime'] = pd.to_datetime(df['Datetime'])
                elif 'Date' in df.columns:
                    # Rename 'Date' to 'Datetime' if it represents specific timestamps
                    df['Datetime'] = pd.to_datetime(df['Date'])
                    df = df.drop(columns=['Date'])

                # Save file
                file_name = f"{ticker.replace('.', '_')}_realtime.csv"
                file_path = os.path.join(self.raw_data_dir, file_name)
                df.to_csv(file_path, index=False)
                
                saved_paths.append(file_path)
                logger.info(f"Successfully saved {ticker} real-time data ({len(df)} rows) to {file_path}")

            except Exception as e:
                logger.error(f"Error fetching real-time data for ticker {ticker}: {str(e)}")

        return saved_paths


if __name__ == "__main__":
    # Test execution
    TICKERS = ["AAPL", "TSLA", "MSFT", "RELIANCE.NS", "TCS.NS"]
    collector = StockDataCollector(raw_data_dir="data/raw")
    
    # Fetch historical
    logger.info("Executing test collection for historical datasets...")
    hist_files = collector.fetch_historical_data(TICKERS, period="5y", interval="1d")
    
    # Fetch real-time (intraday)
    logger.info("Executing test collection for real-time datasets...")
    rt_files = collector.fetch_realtime_data(TICKERS, period="1d", interval="1m")
    
    logger.info(f"Phase 1 - Data Collection complete! Historical: {len(hist_files)} files, Realtime: {len(rt_files)} files.")
