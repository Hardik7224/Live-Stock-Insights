import os
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Dict, List, Any

# Configure logging
logger = logging.getLogger("DatabaseIntegration")

# Hardcoded metadata to seed our dimension table dim_stocks
DIM_STOCKS_SEED: List[Dict[str, str]] = [
    {
        "symbol": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "exchange": "NASDAQ"
    },
    {
        "symbol": "MSFT",
        "company_name": "Microsoft Corporation",
        "sector": "Technology",
        "industry": "Software - Infrastructure",
        "exchange": "NASDAQ"
    },
    {
        "symbol": "TSLA",
        "company_name": "Tesla, Inc.",
        "sector": "Consumer Cyclical",
        "industry": "Auto Manufacturers",
        "exchange": "NASDAQ"
    },
    {
        "symbol": "RELIANCE.NS",
        "company_name": "Reliance Industries Limited",
        "sector": "Energy",
        "industry": "Oil & Gas Refining & Marketing",
        "exchange": "NSE"
    },
    {
        "symbol": "TCS.NS",
        "company_name": "Tata Consultancy Services Limited",
        "sector": "Technology",
        "industry": "Information Technology Services",
        "exchange": "NSE"
    }
]


class StockDatabaseManager:
    """
    Manages SQLAlchemy database connections, schema execution, dim_stocks seeding,
    and bulk price insertions for historical and real-time stock datasets.
    """

    def __init__(self, db_url: str = "sqlite:///data/stock_analytics.db"):
        """
        Initializes the database connection.
        Default: SQLite local file in data/ directory.
        Can be overridden with MySQL/PostgreSQL connection string.
        """
        # Ensure directories exist if using SQLite
        if db_url.startswith("sqlite:///"):
            db_path = db_url.replace("sqlite:///", "")
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
                
        self.engine = create_engine(db_url, echo=False)
        logger.info(f"Connected to database engine with URL: {db_url.split('@')[-1]}") # Redact password if present

    def initialize_schema(self, schema_path: str = "sql/schema.sql") -> bool:
        """
        Reads and executes the SQL schema script to create dim_stocks and fact_stock_prices tables.
        """
        try:
            if not os.path.exists(schema_path):
                logger.error(f"Schema file not found at: {schema_path}")
                return False

            logger.info(f"Initializing database schema from: {schema_path}")
            with open(schema_path, "r") as f:
                schema_sql = f.read()

            # Execute schema in transaction block
            with self.engine.begin() as conn:
                # SQLite doesn't support multiple commands in standard execute() unless separated
                # We split by semicolon to handle SQLite/MySQL cleanly
                queries = [q.strip() for q in schema_sql.split(";") if q.strip()]
                for query in queries:
                    conn.execute(text(query))
                    
            logger.info("Database schema initialized successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize database schema: {str(e)}")
            return False

    def seed_dimension_table(self) -> bool:
        """
        Seeds metadata into the dim_stocks dimension table.
        """
        try:
            logger.info("Seeding dim_stocks table...")
            df_seed = pd.DataFrame(DIM_STOCKS_SEED)
            
            with self.engine.begin() as conn:
                # First delete any existing dimensions to avoid duplicate primary keys
                conn.execute(text("DELETE FROM dim_stocks"))
                # Write seed values
                df_seed.to_sql("dim_stocks", con=conn, if_exists="append", index=False)
                
            logger.info(f"Successfully seeded {len(df_seed)} stocks in dim_stocks.")
            return True
        except Exception as e:
            logger.error(f"Failed to seed dim_stocks: {str(e)}")
            return False

    def insert_processed_prices(self, csv_path: str) -> bool:
        """
        Loads processed CSV files, converts formats, and transactionally updates 
        the fact_stock_prices table. Matches columns dynamically.
        """
        try:
            if not os.path.exists(csv_path):
                logger.error(f"Processed CSV file not found: {csv_path}")
                return False

            df = pd.read_csv(csv_path)
            if df.empty:
                logger.warning(f"CSV {csv_path} has no rows. Skipping insertion.")
                return False

            ticker = df['Ticker'].iloc[0]
            logger.info(f"Loading {len(df)} rows for {ticker} into fact_stock_prices...")

            # Clean and map columns to match the SQL table columns
            # Column mapping dictionary
            col_mapping = {
                'Date': 'date',
                'Datetime': 'date', # For intraday we load datetime into date column
                'Ticker': 'symbol',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume',
                'Daily_Return': 'daily_return',
                'Pct_Change': 'pct_change',
                'Volatility_20d': 'volatility_20d',
                'Volatility_5m': 'volatility_20d', # Intraday volatility mapped to same column
                'SMA_20': 'sma_20',
                'SMA_50': 'sma_50',
                'EMA_20': 'ema_20',
                'RSI_14': 'rsi_14',
                'MACD_Line': 'macd_line',
                'MACD_Signal': 'macd_signal',
                'MACD_Hist': 'macd_hist',
                'BB_Upper': 'bb_upper',
                'BB_Middle': 'bb_middle',
                'BB_Lower': 'bb_lower',
                'BB_Bandwidth': 'bb_bandwidth',
                'Daily_Return_Outlier': 'Daily_Return_Outlier',
                'Volume_Outlier': 'Volume_Outlier'
            }

            # Columns in SQL table to keep
            present_cols = {c: col_mapping[c] for c in df.columns if c in col_mapping}
            df_db = df[list(present_cols.keys())].rename(columns=present_cols)

            # Ensure data types match SQLite expectations
            # (e.g. standard strings for dates, float for decimal numbers)
            df_db['date'] = df_db['date'].astype(str)

            with self.engine.begin() as conn:
                # 1. Transactional Delete of existing rows for this symbol
                # to prevent UNIQUE constraints and duplicates
                conn.execute(
                    text("DELETE FROM fact_stock_prices WHERE symbol = :symbol"), 
                    {"symbol": ticker}
                )
                
                # 2. Insert new records
                df_db.to_sql("fact_stock_prices", con=conn, if_exists="append", index=False)

            logger.info(f"Successfully committed {len(df_db)} records for {ticker} in fact_stock_prices.")
            return True

        except Exception as e:
            logger.error(f"Failed to insert stock prices from {csv_path}: {str(e)}")
            return False

    def run_advanced_queries(self, queries_path: str = "sql/queries.sql") -> Dict[str, pd.DataFrame]:
        """
        Bonus validation function: reads the queries portfolio and runs each query 
        on the database, returning Pandas DataFrames of the results. 
        Verifies database works perfectly!
        """
        results = {}
        try:
            if not os.path.exists(queries_path):
                logger.warning(f"Queries file not found at: {queries_path}")
                return results

            with open(queries_path, "r") as f:
                queries_sql = f.read()

            # Queries are separated by semicolon.
            # We skip comments and empty lines
            import re
            query_blocks = queries_sql.split(";")
            
            # Simple parser to find query titles or execute them sequentially
            query_counter = 1
            for q_raw in query_blocks:
                q = q_raw.strip()
                if not q:
                    continue
                
                # Find query name or title from comments
                title_match = re.search(r'QUERY\s+\d+:\s+([^\n]+)', q)
                if title_match:
                    query_name = title_match.group(1).strip()
                else:
                    query_name = f"Query_{query_counter}"

                try:
                    logger.info(f"Validating SQL portfolio query: '{query_name}'")
                    # Remove SQL comments for execution if SQLite gets confused
                    sql_clean = re.sub(r'--.*$', '', q, flags=re.MULTILINE).strip()
                    if sql_clean:
                        df_res = pd.read_sql_query(sql_clean, con=self.engine)
                        results[query_name] = df_res
                        logger.info(f"Query '{query_name}' returned {len(df_res)} rows.")
                        query_counter += 1
                except Exception as ex:
                    logger.warning(f"Could not execute query '{query_name}': {str(ex)}")

        except Exception as e:
            logger.error(f"Failed to run query analysis: {str(e)}")

        return results


if __name__ == "__main__":
    # Test execution
    manager = StockDatabaseManager()
    manager.initialize_schema()
    manager.seed_dimension_table()
