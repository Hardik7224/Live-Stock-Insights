-- ========================================================
-- ADVANCED SQL ANALYSIS & REPORTING QUERIES
-- ========================================================
-- These queries showcase senior-level SQL concepts including:
-- - Common Table Expressions (CTEs)
-- - Window Functions (LAG, LEAD, DENSE_RANK, AVG OVER)
-- - Complex joins and aggregations
-- - Business metrics calculations (MoM, Rolling Averages, RSI Alerts)
-- ========================================================

-- QUERY 1: CORE STOCK PERFORMANCE SUMMARY (DIMENSION JOIN & AGGREGATIONS)
-- Combines the dimension and fact tables to calculate high-level performance metrics per stock.
SELECT 
    s.symbol,
    s.company_name,
    s.sector,
    COUNT(f.id) AS total_trading_days,
    ROUND(MIN(f.close), 2) AS min_price,
    ROUND(MAX(f.close), 2) AS max_price,
    ROUND(AVG(f.close), 2) AS avg_close_price,
    ROUND(AVG(f.volume), 0) AS avg_daily_volume
FROM fact_stock_prices f
INNER JOIN dim_stocks s ON f.symbol = s.symbol
GROUP BY s.symbol, s.company_name, s.sector
ORDER BY avg_close_price DESC;


-- QUERY 2: MONTH-OVER-MONTH PERFORMANCE TRENDS (CTEs & LEAD/LAG WINDOW FUNCTIONS)
-- Computes the Month-over-Month (MoM) growth percentage of average closing prices for each stock.
WITH MonthlyAverages AS (
    SELECT 
        symbol,
        STRFTIME('%Y-%m', date) AS year_month,
        AVG(close) AS avg_close
    FROM fact_stock_prices
    GROUP BY symbol, year_month
),
MoMAnalysis AS (
    SELECT 
        symbol,
        year_month,
        avg_close,
        LAG(avg_close, 1) OVER (PARTITION BY symbol ORDER BY year_month) AS prev_month_avg_close
    FROM MonthlyAverages
)
SELECT 
    symbol,
    year_month,
    ROUND(avg_close, 2) AS current_month_avg_close,
    ROUND(prev_month_avg_close, 2) AS previous_month_avg_close,
    ROUND(((avg_close - prev_month_avg_close) / prev_month_avg_close) * 100, 2) AS mom_pct_change
FROM MoMAnalysis
WHERE prev_month_avg_close IS NOT NULL
ORDER BY symbol, year_month;


-- QUERY 3: PURE SQL 20-DAY AND 50-DAY SIMPLE MOVING AVERAGES (ROLLING WINDOW FUNCTIONS)
-- Re-calculates Simple Moving Averages (SMA) directly in the database engine using Window frames.
-- Highly useful for cross-validating Pandas calculations or running database-native alerts.
SELECT 
    date,
    symbol,
    close,
    ROUND(AVG(close) OVER (
        PARTITION BY symbol 
        ORDER BY date 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ), 2) AS db_calculated_sma_20,
    ROUND(AVG(close) OVER (
        PARTITION BY symbol 
        ORDER BY date 
        ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
    ), 2) AS db_calculated_sma_50
FROM fact_stock_prices
ORDER BY symbol, date DESC
LIMIT 100;


-- QUERY 4: IDENTIFY OVERBOUGHT & OVERSOLD SIGNAL OPPORTUNITIES (FINANCIAL ALERTS)
-- Pinpoints dates where stocks crossed momentum thresholds (RSI < 30 oversold/buy or > 70 overbought/sell).
SELECT 
    f.date,
    s.company_name,
    f.symbol,
    ROUND(f.close, 2) AS closing_price,
    ROUND(f.rsi_14, 2) AS rsi_14,
    CASE 
        WHEN f.rsi_14 >= 70 THEN 'OVERBOUGHT (SELL ALERT)'
        WHEN f.rsi_14 <= 30 THEN 'OVERSOLD (BUY OPPORTUNITY)'
        ELSE 'NEUTRAL'
    END AS signal_status,
    ROUND(f.pct_change, 2) AS daily_pct_change
FROM fact_stock_prices f
INNER JOIN dim_stocks s ON f.symbol = s.symbol
WHERE f.rsi_14 >= 70 OR f.rsi_14 <= 30
ORDER BY f.date DESC, f.rsi_14 DESC
LIMIT 50;


-- QUERY 5: IDENTIFY TOP 5 TRADING DAYS BY TRADING VOLUME PER YEAR (RANKING FUNCTIONS)
-- Demonstrates partition ranking using DENSE_RANK() to find highest trading volumes.
WITH RankedVolumes AS (
    SELECT 
        STRFTIME('%Y', date) AS trading_year,
        date,
        symbol,
        volume,
        close,
        DENSE_RANK() OVER (
            PARTITION BY STRFTIME('%Y', date), symbol 
            ORDER BY volume DESC
        ) as volume_rank
    FROM fact_stock_prices
)
SELECT 
    trading_year,
    date,
    symbol,
    volume,
    ROUND(close, 2) AS closing_price,
    volume_rank
FROM RankedVolumes
WHERE volume_rank <= 5
ORDER BY symbol, trading_year DESC, volume_rank;
