import os
import sys
import logging
import pandas as pd

# Standardize path imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.data_collection import StockDataCollector
from scripts.data_preprocessing import StockDataPreprocessor
from scripts.db_integration import StockDatabaseManager

# Set up logging for master pipeline
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (MasterETL): %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("MasterPipeline")

# Defined stocks list
TICKERS = ["AAPL", "TSLA", "MSFT", "RELIANCE.NS", "TCS.NS"]


def run_etl_pipeline(db_url: str = "sqlite:///data/stock_analytics.db"):
    """
    Orchestrates the entire End-to-End ETL Pipeline.
    """
    logger.info("==============================================================")
    logger.info("STARTING END-TO-END STOCK ANALYTICS ETL PIPELINE")
    logger.info("==============================================================")

    # -------------------------------------------------------------------------
    # PHASE 1: DATA COLLECTION
    # -------------------------------------------------------------------------
    logger.info("\n--- PHASE 1: EXTRACTING DATA FROM YFINANCE API ---")
    collector = StockDataCollector(raw_data_dir="data/raw")
    
    # 1.1 Historical daily data
    logger.info(f"Fetching 5-Year Historical Daily Bars for: {TICKERS}")
    raw_hist_files = collector.fetch_historical_data(TICKERS, period="5y", interval="1d")
    logger.info(f"Successfully downloaded {len(raw_hist_files)} raw historical files.")

    # 1.2 Real-time intraday data (1-minute intervals for today)
    logger.info(f"Fetching Intraday 1-Minute Bars for: {TICKERS}")
    raw_rt_files = collector.fetch_realtime_data(TICKERS, period="1d", interval="1m")
    logger.info(f"Successfully downloaded {len(raw_rt_files)} raw real-time files.")

    if not raw_hist_files:
        logger.error("No historical datasets collected. ETL Pipeline aborted.")
        return False

    # -------------------------------------------------------------------------
    # PHASE 2: DATA PREPROCESSING & FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    logger.info("\n--- PHASE 2: TRANSFORMING DATA & QUANTITATIVE FEATURE ENGINEERING ---")
    preprocessor = StockDataPreprocessor(raw_data_dir="data/raw", processed_data_dir="data/processed")
    
    processed_hist_files = []
    for file_path in raw_hist_files:
        out_path = preprocessor.preprocess_historical_file(file_path)
        if out_path:
            processed_hist_files.append(out_path)
            
    processed_rt_files = []
    for file_path in raw_rt_files:
        out_path = preprocessor.preprocess_realtime_file(file_path)
        if out_path:
            processed_rt_files.append(out_path)
            
    logger.info(f"Transformation complete! Preprocessed: {len(processed_hist_files)} Historical, {len(processed_rt_files)} Real-time.")

    # -------------------------------------------------------------------------
    # PHASE 3: SQL DATABASE LOADING
    # -------------------------------------------------------------------------
    logger.info("\n--- PHASE 3: LOADING DATA INTO SQL DATABASE ---")
    db_manager = StockDatabaseManager(db_url=db_url)
    
    # 3.1 Initialize schema
    if not db_manager.initialize_schema(schema_path="sql/schema.sql"):
        logger.error("Database schema initialization failed. SQL load aborted.")
        return False
        
    # 3.2 Seed stock dimensions
    if not db_manager.seed_dimension_table():
        logger.error("Dimension table seeding failed. SQL load aborted.")
        return False

    # 3.3 Load historical pricing facts
    logger.info("Loading preprocessed historical price facts...")
    for hist_csv in processed_hist_files:
        db_manager.insert_processed_prices(hist_csv)

    # 3.4 Load real-time pricing facts (overwrites/appends intraday)
    logger.info("Loading preprocessed real-time intraday price facts...")
    for rt_csv in processed_rt_files:
        db_manager.insert_processed_prices(rt_csv)

    logger.info("Database load operations complete.")

    # -------------------------------------------------------------------------
    # PHASE 4: VALIDATING ADVANCED SQL PORFOLIO
    # -------------------------------------------------------------------------
    logger.info("\n--- PHASE 4: RUNNING PORTFOLIO SQL QUERIES AS VALIDATION ---")
    query_results = db_manager.run_advanced_queries(queries_path="sql/queries.sql")
    
    logger.info("Advanced SQL queries executed successfully. Output summary:")
    for query_name, df_result in query_results.items():
        logger.info(f"\n>>> Query: '{query_name}' (First 3 rows shown):")
        print(df_result.head(3).to_string(index=False))
        print("-" * 60)

    logger.info("==============================================================")
    logger.info("END-TO-END STOCK ANALYTICS ETL PIPELINE EXECUTED SUCCESSFULLY")
    logger.info("==============================================================")
    return True


if __name__ == "__main__":
    run_etl_pipeline()
