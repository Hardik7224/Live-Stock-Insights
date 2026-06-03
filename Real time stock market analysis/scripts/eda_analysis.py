import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logger = logging.getLogger("ExploratoryDataAnalysis")


class StockEDA:
    """
    A class to perform programmatic Exploratory Data Analysis (EDA) on preprocessed 
    stock data and generate production-grade visualizations.
    """

    def __init__(self, processed_data_dir: str = "data/processed", image_save_dir: str = "images"):
        self.processed_data_dir = processed_data_dir
        self.image_save_dir = image_save_dir
        os.makedirs(self.image_save_dir, exist_ok=True)
        
        # Configure seaborn premium style
        sns.set_theme(style="darkgrid")
        plt.rcParams.update({
            "figure.autolayout": True,
            "figure.figsize": (10, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.dpi": 200
        })
        logger.info(f"Initialized StockEDA. Visualizations will be saved to: {self.image_save_dir}")

    def load_combined_historical_data(self, tickers: list[str]) -> pd.DataFrame:
        """
        Combines processed CSVs for all tickers into a single DataFrame.
        """
        all_dfs = []
        for ticker in tickers:
            file_name = f"{ticker.replace('.', '_')}_historical_processed.csv"
            file_path = os.path.join(self.processed_data_dir, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['Date'])
                all_dfs.append(df)
            else:
                logger.warning(f"No processed data found for {ticker} at {file_path}")
        
        if not all_dfs:
            raise ValueError("No stock files loaded successfully.")
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df

    def plot_stock_trends(self, combined_df: pd.DataFrame):
        """
        Generates normalized cumulative growth comparison line charts.
        Normalized to 100 on day 1 to show relative growth.
        """
        logger.info("Generating stock trend analysis...")
        plt.figure()
        
        # Pivot to get Dates as index and Tickers as columns
        pivot_df = combined_df.pivot(index='Date', columns='Ticker', values='Close').sort_index()
        
        # Interpolate missing values in case of differing holiday schedules between NASDAQ and NSE
        pivot_df = pivot_df.ffill().bfill()
        
        # Normalize to base 100
        normalized_df = (pivot_df / pivot_df.iloc[0]) * 100
        
        for col in normalized_df.columns:
            plt.plot(normalized_df.index, normalized_df[col], label=col, linewidth=1.5)
            
        plt.title("Normalized Historical Stock Price Trends (Base = 100)")
        plt.xlabel("Timeline")
        plt.ylabel("Relative Growth (%)")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.5)
        
        save_path = os.path.join(self.image_save_dir, "stock_trends_comparison.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved stock trends line chart to {save_path}")

    def plot_correlation_matrix(self, combined_df: pd.DataFrame):
        """
        Generates correlation heatmap of closing prices.
        """
        logger.info("Generating stock correlation heatmap...")
        plt.figure(figsize=(8, 6))
        
        pivot_df = combined_df.pivot(index='Date', columns='Ticker', values='Close').sort_index().ffill().bfill()
        corr_matrix = pivot_df.corr()
        
        # Plot Heatmap with high-quality colors
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap="coolwarm", 
            vmin=-1, 
            vmax=1, 
            fmt=".2f", 
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        
        plt.title("Closing Price Correlation Heatmap")
        
        save_path = os.path.join(self.image_save_dir, "stock_correlation_heatmap.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved correlation heatmap to {save_path}")

    def plot_volatility_boxplot(self, combined_df: pd.DataFrame):
        """
        Generates return distribution boxplots for risk profiling.
        """
        logger.info("Generating daily return volatility boxplots...")
        plt.figure()
        
        # Filter returns within standard thresholds for cleaner visual focus
        # Extreme market outliers can distort the box scale
        filtered_df = combined_df[(combined_df['Daily_Return'] > -0.1) & (combined_df['Daily_Return'] < 0.1)]
        
        sns.boxplot(x='Ticker', y='Daily_Return', data=filtered_df, palette="Set2")
        
        plt.title("Daily Returns Volatility Distribution (Outliers Clipped at ±10%)")
        plt.xlabel("Stock Ticker")
        plt.ylabel("Daily Return")
        
        save_path = os.path.join(self.image_save_dir, "stock_volatility_boxplot.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved volatility boxplot to {save_path}")

    def plot_volume_analysis(self, combined_df: pd.DataFrame):
        """
        Generates relative volume traded comparisons.
        """
        logger.info("Generating average volume analysis...")
        plt.figure()
        
        avg_volume = combined_df.groupby('Ticker')['Volume'].mean().reset_index()
        
        # Scale volume to millions for better label scaling
        avg_volume['Volume_Millions'] = avg_volume['Volume'] / 1e6
        
        sns.barplot(x='Ticker', y='Volume_Millions', data=avg_volume, palette="viridis")
        
        plt.title("Average Daily Trading Volume (in Millions of Shares)")
        plt.xlabel("Stock Ticker")
        plt.ylabel("Volume (Millions)")
        plt.grid(axis='y', linestyle="--", alpha=0.5)
        
        save_path = os.path.join(self.image_save_dir, "stock_volume_comparison.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved volume analysis bar chart to {save_path}")

    def plot_return_distributions(self, combined_df: pd.DataFrame):
        """
        Generates histogram distribution curves for ticker returns.
        """
        logger.info("Generating daily return histograms...")
        plt.figure()
        
        # We will plot AAPL and TSLA for comparison (Stable tech vs volatile EV)
        for ticker in ['AAPL', 'TSLA']:
            ticker_df = combined_df[combined_df['Ticker'] == ticker]
            sns.kdeplot(ticker_df['Daily_Return'], fill=True, label=ticker, alpha=0.4)
            
        plt.title("Daily Return Distribution Probability Curve (AAPL vs TSLA)")
        plt.xlabel("Daily Return")
        plt.ylabel("Density")
        plt.xlim(-0.08, 0.08) # Focus zoom on main return zones
        plt.legend()
        
        save_path = os.path.join(self.image_save_dir, "stock_return_distribution.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved return distribution curve to {save_path}")

    def run_all_eda(self, tickers: list[str]) -> bool:
        """
        Runs the complete EDA suite and saves all figures.
        """
        try:
            logger.info("Starting EDA run...")
            df = self.load_combined_historical_data(tickers)
            
            self.plot_stock_trends(df)
            self.plot_correlation_matrix(df)
            self.plot_volatility_boxplot(df)
            self.plot_volume_analysis(df)
            self.plot_return_distributions(df)
            
            logger.info("All EDA plots generated successfully.")
            return True
        except Exception as e:
            logger.error(f"EDA Generation failed: {str(e)}")
            return False


if __name__ == "__main__":
    TICKERS = ["AAPL", "TSLA", "MSFT", "RELIANCE.NS", "TCS.NS"]
    eda = StockEDA()
    eda.run_all_eda(TICKERS)
