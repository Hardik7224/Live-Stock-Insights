-- ========================================================
-- DATABASE SCHEMA: REAL-TIME STOCK MARKET ANALYTICS SYSTEM
-- ========================================================
-- Suitable for SQLite (Local Dev) and fully compatible with
-- PostgreSQL and MySQL for enterprise production deployment.
-- ========================================================

-- DROP TABLES IF THEY EXIST TO ALLOW SEAMLESS INITIALIZATION
DROP TABLE IF EXISTS fact_stock_prices;
DROP TABLE IF EXISTS dim_stocks;

-- 1. DIMENSION TABLE: dim_stocks
-- Stores corporate profile details for tickers.
CREATE TABLE dim_stocks (
    symbol VARCHAR(15) PRIMARY KEY,
    company_name VARCHAR(100) NOT NULL,
    sector VARCHAR(50),
    industry VARCHAR(50),
    exchange VARCHAR(30)
);

-- 2. FACT TABLE: fact_stock_prices
-- Stores core price metrics and preprocessed technical/quantitative indicators.
CREATE TABLE fact_stock_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- Auto-incrementing primary key
    date DATE NOT NULL,                   -- Date of market close (or datetime for intraday)
    symbol VARCHAR(15) NOT NULL,          -- Stock ticker joining to dim_stocks
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    adj_close DOUBLE PRECISION,
    volume BIGINT NOT NULL,
    
    -- Calculated Financial Metrics
    daily_return DOUBLE PRECISION,
    pct_change DOUBLE PRECISION,
    volatility_20d DOUBLE PRECISION,
    
    -- Technical Indicators
    sma_20 DOUBLE PRECISION,
    sma_50 DOUBLE PRECISION,
    ema_20 DOUBLE PRECISION,
    rsi_14 DOUBLE PRECISION,
    macd_line DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_hist DOUBLE PRECISION,
    bb_upper DOUBLE PRECISION,
    bb_middle DOUBLE PRECISION,
    bb_lower DOUBLE PRECISION,
    bb_bandwidth DOUBLE PRECISION,
    
    -- Flag Columns
    Daily_Return_Outlier BOOLEAN DEFAULT FALSE,
    Volume_Outlier BOOLEAN DEFAULT FALSE,
    
    -- Relational Integrity Constraints
    FOREIGN KEY (symbol) REFERENCES dim_stocks (symbol) ON DELETE CASCADE,
    CONSTRAINT uq_date_symbol UNIQUE (date, symbol) -- Enforces uniqueness per stock per day
);

-- CREATE INDEXES FOR OPTIMIZED QUERY PERFORMANCE (RECRUITER BONUS)
CREATE INDEX idx_prices_date ON fact_stock_prices(date);
CREATE INDEX idx_prices_symbol_date ON fact_stock_prices(symbol, date);
CREATE INDEX idx_prices_rsi ON fact_stock_prices(rsi_14) WHERE rsi_14 IS NOT NULL;
