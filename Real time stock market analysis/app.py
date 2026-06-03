import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sqlalchemy import create_engine, text

# Add workspace to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page Configuration for Wide Layout & Dark Mode Vibes
st.set_page_config(
    page_title="QUANTUM | Real-Time Stock Analytics & Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling (Glassmorphism & High Contrast Dark Mode)
st.markdown("""
    <style>
    /* Premium Title and Brand */
    .brand-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .brand-subtitle {
        font-size: 1rem;
        color: #a0aec0;
        margin-bottom: 2rem;
    }
    /* Metric Card Customizations */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #ffffff;
    }
    div[data-testid="stMetricLabel"] {
        color: #a0aec0 !important;
        font-weight: 500;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #1a202c;
    }
    /* Status indicators */
    .metric-container {
        background: rgba(26, 32, 44, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Database Engine
DB_URL = "sqlite:///data/stock_analytics.db"
engine = create_engine(DB_URL)

# Hardcoded stocks descriptions
STOCK_PROFILES = {
    "AAPL": {"name": "Apple Inc.", "exchange": "NASDAQ", "sector": "Technology", "industry": "Consumer Electronics"},
    "MSFT": {"name": "Microsoft Corporation", "exchange": "NASDAQ", "sector": "Technology", "industry": "Software - Infrastructure"},
    "TSLA": {"name": "Tesla, Inc.", "exchange": "NASDAQ", "sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
    "RELIANCE.NS": {"name": "Reliance Industries Limited", "exchange": "NSE", "sector": "Energy", "industry": "Oil & Gas Refining"},
    "TCS.NS": {"name": "Tata Consultancy Services Limited", "exchange": "NSE", "sector": "Technology", "industry": "IT Services"}
}

# =========================================================================
# UTILITIES & BOOTSTRAPPING
# =========================================================================
def check_pipeline_assets() -> bool:
    """Checks if the data and model files exist."""
    required_paths = [
        "data/stock_analytics.db",
        "data/processed/AAPL_historical_processed.csv"
    ]
    return all(os.path.exists(p) for p in required_paths)

def bootstrap_pipeline():
    """Runs the master ETL and Machine Learning sequentially from the UI."""
    try:
        from main import run_etl_pipeline
        from scripts.ml_pipeline import StockMLPipeline
        
        # 1. Run ETL
        with st.spinner("Extracting yfinance data, generating quantitative indicators, and loading SQL DB..."):
            success = run_etl_pipeline(db_url=DB_URL)
            if not success:
                st.error("ETL pipeline failed during extraction or database seeding.")
                return False
                
        # 2. Run ML
        with st.spinner("Training Random Forest, Gradient Boosting, and Linear Regression models for all tickers..."):
            pipeline = StockMLPipeline(processed_data_dir="data/processed", model_save_dir="models/saved_models")
            pipeline.train_all_tickers(list(STOCK_PROFILES.keys()))
            
        # 3. Generate EDA images
        with st.spinner("Creating static exploratory graphics..."):
            from scripts.eda_analysis import StockEDA
            eda = StockEDA(processed_data_dir="data/processed", image_save_dir="images")
            eda.run_all_eda(list(STOCK_PROFILES.keys()))

        st.success("Platform initialized successfully! Enjoy full exploration.")
        return True
    except Exception as e:
        st.error(f"Bootstrapping failed: {str(e)}")
        return False

# =========================================================================
# APP SIDEBAR & NAVIGATION
# =========================================================================
st.sidebar.markdown("<h1 style='color: #00f2fe;'>QUANTUM ANALYTICS</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 0.8rem; color: #a0aec0; margin-top: -10px;'>End-to-End Stock Analytics & Prediction</p>", unsafe_allow_html=True)

# 1. Navigation Panel
page = st.sidebar.radio("Go to Page:", ["📈 Market Overview & Charts", "💻 SQL Database Analytics", "🔮 Quantitative ML Forecasting", "💼 Portfolio Optimizer"])

# 2. Global Ticker Selector
selected_ticker = st.sidebar.selectbox("Select Asset Ticker:", list(STOCK_PROFILES.keys()))
profile = STOCK_PROFILES[selected_ticker]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Asset Profile:**")
st.sidebar.markdown(f"- **Company:** {profile['name']}")
st.sidebar.markdown(f"- **Exchange:** {profile['exchange']}")
st.sidebar.markdown(f"- **Sector:** {profile['sector']}")
st.sidebar.markdown(f"- **Industry:** {profile['industry']}")

# Check database
assets_exist = check_pipeline_assets()
if not assets_exist:
    st.sidebar.warning("⚠️ Database or processed data not found.")
    if st.sidebar.button("Bootstrap Platform 🚀", use_container_width=True):
        if bootstrap_pipeline():
            st.rerun()
    st.info("Please click the bootstrap button to download data and train models automatically!")
    st.stop()

# Load Selected Stock Data from SQL (Recruiter demonstration of DB connector)
@st.cache_data(ttl=60)
def load_stock_data_from_db(ticker: str) -> pd.DataFrame:
    try:
        query = f"SELECT * FROM fact_stock_prices WHERE symbol = '{ticker}' ORDER BY date ASC"
        df = pd.read_sql_query(query, con=engine)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Failed to query database: {str(e)}")
        return pd.DataFrame()

df_prices = load_stock_data_from_db(selected_ticker)

if df_prices.empty:
    st.warning("Database holds empty data for this ticker. Run the master pipeline first.")
    st.stop()

# Date range selection on the loaded data
min_date = df_prices['date'].min().to_pydatetime()
max_date = df_prices['date'].max().to_pydatetime()
start_date, end_date = st.sidebar.slider(
    "Timeline Selector:",
    min_value=min_date,
    max_value=max_date,
    value=(df_prices['date'].iloc[-365].to_pydatetime(), max_date) # Default last 1 year
)

# Filter dataset
df_filtered = df_prices[(df_prices['date'] >= start_date) & (df_prices['date'] <= end_date)].reset_index(drop=True)

# =========================================================================
# PAGE 1: MARKET OVERVIEW & CHARTS
# =========================================================================
if page == "📈 Market Overview & Charts":
    st.markdown(f"<div class='brand-title'>{profile['name']} ({selected_ticker})</div>", unsafe_allow_html=True)
    st.markdown("<div class='brand-subtitle'>Real-Time Dynamic Interactive Visualizer & Quantitative Overlays</div>", unsafe_allow_html=True)
    
    # 1. KPI CARDS ROW
    # Calculate latest rates
    latest_row = df_prices.iloc[-1]
    prev_row = df_prices.iloc[-2]
    
    current_price = latest_row['close']
    price_change = latest_row['pct_change']
    price_diff = current_price - prev_row['close']
    high_52 = df_prices['high'].tail(252).max()
    low_52 = df_prices['low'].tail(252).min()
    volume_latest = latest_row['volume']

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric(
        label="Current Close Price", 
        value=f"${current_price:.2f}" if "NS" not in selected_ticker else f"₹{current_price:.2f}",
        delta=f"{price_diff:+.2f} ({price_change:+.2f}%)"
    )
    kpi2.metric(
        label="52-Week High", 
        value=f"${high_52:.2f}" if "NS" not in selected_ticker else f"₹{high_52:.2f}"
    )
    kpi3.metric(
        label="52-Week Low", 
        value=f"${low_52:.2f}" if "NS" not in selected_ticker else f"₹{low_52:.2f}"
    )
    kpi4.metric(
        label="Daily Trading Volume", 
        value=f"{volume_latest:,.0f} shares"
    )
    # Check outlier status
    is_vol_outlier = latest_row['Volume_Outlier']
    kpi5.metric(
        label="Volume Anomaly State",
        value="⚠️ ANOMALOUS" if is_vol_outlier else "✅ NORMAL",
        delta="Extreme Activity" if is_vol_outlier else "Standard Traded Range",
        delta_color="inverse" if is_vol_outlier else "normal"
    )

    st.markdown("---")

    # 2. CHART LAYOUT CONTROLS
    st.markdown("### Technical Indicator Controls")
    c1, c2, c3, c4 = st.columns(4)
    show_ma = c1.checkbox("Simple Moving Averages (20 & 50 SMA)", value=True)
    show_ema = c2.checkbox("Exponential Moving Average (20 EMA)", value=False)
    show_bb = c3.checkbox("Bollinger Bands (20 Day, 2.0 SD)", value=False)
    sub_chart = c4.selectbox("Sub-Chart Oscillator:", ["None", "RSI (Relative Strength Index)", "MACD (Trend Momentum)"])

    # 3. INTERACTIVE PLOTLY CHART
    # We will build a dual-panel chart: Upper panel is Candlestick, Lower is Volume or Oscillator
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08, 
        row_heights=[0.75, 0.25]
    )

    # 3.1 Candlestick Chart
    fig.add_trace(
        go.Candlestick(
            x=df_filtered['date'],
            open=df_filtered['open'],
            high=df_filtered['high'],
            low=df_filtered['low'],
            close=df_filtered['close'],
            name="Stock Price",
            increasing_line_color='#26a69a', 
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )

    # 3.2 Overlays
    if show_ma:
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['sma_20'], name='20 SMA', line=dict(color='#ff9800', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['sma_50'], name='50 SMA', line=dict(color='#9c27b0', width=1.5)), row=1, col=1)
        
    if show_ema:
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['ema_20'], name='20 EMA', line=dict(color='#00e676', width=1.5)), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['bb_upper'], name='Bollinger Upper', line=dict(color='rgba(173, 216, 230, 0.6)', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['bb_lower'], name='Bollinger Lower', line=dict(color='rgba(173, 216, 230, 0.6)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.05)'), row=1, col=1)

    # 3.3 Sub-charts (Oscillators)
    if sub_chart == "None":
        # Default subchart is volume
        fig.add_trace(
            go.Bar(
                x=df_filtered['date'], 
                y=df_filtered['volume'], 
                name='Volume',
                marker_color=np.where(df_filtered['close'] > df_filtered['open'], '#26a69a', '#ef5350')
            ), 
            row=2, col=1
        )
        fig.update_yaxes(title_text="Traded Volume", row=2, col=1)
    elif sub_chart == "RSI (Relative Strength Index)":
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['rsi_14'], name='RSI (14)', line=dict(color='#e040fb', width=1.5)), row=2, col=1)
        # Threshold lines at 30 and 70
        fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", annotation_text="Overbought (70)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", annotation_text="Oversold (30)", row=2, col=1)
        fig.update_yaxes(title_text="RSI Value", range=[10, 90], row=2, col=1)
    elif sub_chart == "MACD (Trend Momentum)":
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['macd_line'], name='MACD Line', line=dict(color='#29b6f6', width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['macd_signal'], name='Signal Line', line=dict(color='#ffca28', width=1.5)), row=2, col=1)
        fig.add_trace(
            go.Bar(
                x=df_filtered['date'], 
                y=df_filtered['macd_hist'], 
                name='MACD Hist',
                marker_color=np.where(df_filtered['macd_hist'] > 0, '#26a69a', '#ef5350')
            ), 
            row=2, col=1
        )
        fig.update_yaxes(title_text="MACD Divergence", row=2, col=1)

    # Clean styling
    fig.update_layout(
        height=650,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_rangeslider_visible=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(gridcolor='rgba(255, 255, 255, 0.05)', showspikes=True)
    fig.update_yaxes(gridcolor='rgba(255, 255, 255, 0.05)', row=1, col=1)
    fig.update_yaxes(gridcolor='rgba(255, 255, 255, 0.05)', row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # 4. BUSINESS INSIGHT SECTION
    st.markdown("### 📊 Business Intelligence & Technical Insights")
    col_ins1, col_ins2 = st.columns(2)
    
    with col_ins1:
        st.markdown(f"**Current Trend Momentum Analysis:**")
        # Generate dynamic financial text based on SMA / RSI values
        rsi_now = latest_row['rsi_14']
        close_now = latest_row['close']
        sma_20_now = latest_row['sma_20']
        
        if rsi_now >= 70:
            rsi_text = f"The Relative Strength Index (RSI) is at **{rsi_now:.2f}** which exceeds the overbought threshold of 70. The asset is technically overextended, implying a potential cooling-off period or localized selling pressure."
        elif rsi_now <= 30:
            rsi_text = f"The Relative Strength Index (RSI) is at **{rsi_now:.2f}** which lies below the oversold threshold of 30. This denotes heavy capitulation and technical undervaluation, frequently marking a buying opportunity for reversion."
        else:
            rsi_text = f"The Relative Strength Index (RSI) is resting at a healthy **{rsi_now:.2f}** index, indicating balanced buying and selling momentum with no immediate overbought or oversold extremes."

        st.write(rsi_text)

    with col_ins2:
        st.markdown("**Moving Average Crossover Status:**")
        if close_now > sma_20_now:
            ma_text = f"The current price (${close_now:.2f}) is trading **above its 20-day Simple Moving Average** (${sma_20_now:.2f}). This exhibits positive short-term bullish momentum, indicating that buyers are maintaining control."
        else:
            ma_text = f"The current price (${close_now:.2f}) is trading **below its 20-day Simple Moving Average** (${sma_20_now:.2f}). This indicates short-term bearish dominance, suggesting a cautious or defensive market stance."
            
        st.write(ma_text)

# =========================================================================
# PAGE 2: SQL DATABASE ANALYTICS (RECRUITER SHOWCASE)
# =========================================================================
elif page == "💻 SQL Database Analytics":
    st.markdown("<div class='brand-title'>Database Portfolio Analytics Hub</div>", unsafe_allow_html=True)
    st.markdown("<div class='brand-subtitle'>Execute Professional Star-Schema Joins & Complex Analytical CTEs directly in the SQLite Engine</div>", unsafe_allow_html=True)
    
    st.markdown("""
    This section connects directly to `data/stock_analytics.db` via SQLAlchemy. Below are five pre-compiled 
    industry-level analytical queries written for this project schema. Select any query to run it live 
    against the database.
    """)

    query_options = {
        "1. Core Stock Performance Summary (Dimension Join & Aggregations)": """
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
        """,
        "2. Month-over-Month Performance Trends (CTEs & LAG Window Function)": """
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
        """,
        "3. Live SQL 20-Day & 50-Day Rolling Moving Averages (Analytical Window Over)": """
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
            LIMIT 50;
        """,
        "4. Momentum Overbought & Oversold Signal Alerts (RSI Trigger Filtering)": """
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
        """,
        "5. Top 5 Trading Volume Days Per Ticker (DENSE_RANK Partitioning)": """
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
        """
    }

    selected_query_label = st.selectbox("Select SQL Query to Execute:", list(query_options.keys()))
    selected_sql = query_options[selected_query_label]

    # Show query syntax
    st.markdown("**SQL Syntax Preview:**")
    st.code(selected_sql, language="sql")

    if st.button("Run Live Query 🚀", use_container_width=True):
        try:
            with engine.connect() as conn:
                df_res = pd.read_sql_query(text(selected_sql), con=conn)
            
            st.success(f"Execution complete! Query returned {len(df_res)} rows.")
            st.dataframe(df_res, use_container_width=True)
            
            # Download capability
            csv_data = df_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Query Result to CSV",
                data=csv_data,
                file_name=f"{selected_query_label.split(' ')[1].lower()}_output.csv",
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"SQL Execution Error: {str(e)}")

# =========================================================================
# PAGE 3: QUANTITATIVE ML FORECASTING
# =========================================================================
elif page == "🔮 Quantitative ML Forecasting":
    st.markdown(f"<div class='brand-title'>{profile['name']} ({selected_ticker}) Forecasting</div>", unsafe_allow_html=True)
    st.markdown("<div class='brand-subtitle'>Next-Day Closing Price Prediction & Model Backtesting Performance Evaluation</div>", unsafe_allow_html=True)

    # Paths
    ticker_clean = selected_ticker.replace('.', '_')
    model_path = f"models/saved_models/{ticker_clean}_best_model.pkl"
    vector_path = f"models/saved_models/{ticker_clean}_latest_vector.pkl"
    features_path = f"models/saved_models/{ticker_clean}_feature_cols.pkl"

    # Check existence
    if not (os.path.exists(model_path) and os.path.exists(vector_path)):
        st.warning("⚠️ Machine learning models for this ticker have not been initialized yet. Train them in the sidebar or run the ML pipeline.")
        st.stop()

    # Load serialized binaries
    best_model = joblib.load(model_path)
    X_latest = joblib.load(vector_path)
    feature_cols = joblib.load(features_path)

    # 1. FORECASTING THE FUTURE (TOMORROW'S CLOSE)
    # Perform prediction
    try:
        # Align features to ensure correct prediction columns
        X_latest_aligned = X_latest[feature_cols]
        predicted_close = best_model.predict(X_latest_aligned)[0]
        
        # Display forecast card
        last_date = df_prices['date'].iloc[-1].strftime('%A, %b %d, %Y')
        
        st.markdown(f"### 🔮 Forward Inference Model")
        
        c_for1, c_for2 = st.columns([2, 1])
        
        with c_for1:
            st.markdown(f"""
            Based on the comprehensive quantitative features recorded as of the market close on **{last_date}**,
            our pre-trained **{type(best_model).__name__}** model has computed a forward-looking prediction 
            for the next trading session's Close price.
            """)
            
            # Show actual last close vs predicted close
            actual_last = df_prices['close'].iloc[-1]
            diff = predicted_close - actual_last
            pct_diff = (diff / actual_last) * 100
            
            st.markdown("---")
            col_met1, col_met2 = st.columns(2)
            col_met1.metric(
                label=f"Last Actual Close Price ({df_prices['date'].iloc[-1].strftime('%m/%d')})", 
                value=f"${actual_last:.2f}" if "NS" not in selected_ticker else f"₹{actual_last:.2f}"
            )
            col_met2.metric(
                label="Predicted Tomorrow's Close Price", 
                value=f"${predicted_close:.2f}" if "NS" not in selected_ticker else f"₹{predicted_close:.2f}",
                delta=f"{diff:+.2f} ({pct_diff:+.2f}%)"
            )
            
        with c_for2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.markdown("**Core Forecasting Model**")
            st.markdown(f"<h3 style='color: #00f2fe; margin-top: 5px;'>{type(best_model).__name__.replace('Regressor', '')}</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.8rem; color: #a0aec0;'>Determined as the best model for this asset based on rigorous chronologically isolated cross-validation tests.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to generate prediction: {str(e)}")

    st.markdown("---")

    # 2. MODEL EVALUATION BACKTEST CHART
    st.markdown("### 📊 Historical Model Backtesting vs Actual Prices")
    
    # To demonstrate backtest fitting, we generate predictions for the entire filtered timeline
    # We rebuild features in real-time or simulate backtest fits
    try:
        from scripts.ml_pipeline import StockMLPipeline
        pipeline = StockMLPipeline()
        
        # Load raw data, extract features
        X_all, y_all, _ = pipeline.prepare_time_series_features(df_filtered)
        X_all_aligned = X_all[feature_cols]
        
        # Generate historical predictions
        predictions = best_model.predict(X_all_aligned)
        
        # Re-attach dates for plotting
        plot_df = pd.DataFrame({
            'Date': df_filtered['date'].iloc[len(df_filtered) - len(predictions):],
            'Actual': y_all,
            'Predicted': predictions
        })

        fig_ml = go.Figure()
        fig_ml.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Actual'], name='Actual Next-Day Close', line=dict(color='#26a69a', width=2)))
        fig_ml.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Predicted'], name='Model Forecasted Next-Day Close', line=dict(color='#ff9800', width=1.5, dash='dash')))

        fig_ml.update_layout(
            height=450,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_ml.update_xaxes(gridcolor='rgba(255, 255, 255, 0.05)')
        fig_ml.update_yaxes(gridcolor='rgba(255, 255, 255, 0.05)', title_text="Price")

        st.plotly_chart(fig_ml, use_container_width=True)

        # 3. EXPLAIN FEATURE IMPORTANCES (IF RANDOM FOREST / GRADIENT BOOSTING)
        if hasattr(best_model, 'feature_importances_'):
            st.markdown("### 🧬 Machine Learning Feature Importances")
            st.write("Understand which quantitative drivers our model prioritized when forecasting closing prices:")
            
            importances = best_model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(10) # Show top 10

            fig_imp = go.Figure(go.Bar(
                x=feat_imp_df['Importance'],
                y=feat_imp_df['Feature'],
                orientation='h',
                marker_color='#4facfe'
            ))
            fig_imp.update_layout(
                height=350,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(autorange="reversed")
            )
            fig_imp.update_xaxes(title_text="Relative Feature Importance Score")
            st.plotly_chart(fig_imp, use_container_width=True)

    except Exception as e:
        st.info("Feature importance or full timeline alignment is loading. Train models to display complete curves.")

# =========================================================================
# PAGE 4: PORTFOLIO OPTIMIZER SIMULATION
# =========================================================================
elif page == "💼 Portfolio Optimizer":
    st.markdown("<div class='brand-title'>Strategic Portfolio Optimizer</div>", unsafe_allow_html=True)
    st.markdown("<div class='brand-subtitle'>Simulate Asset Allocation, Volatility Profiles, and Expected Sharpe Ratios</div>", unsafe_allow_html=True)
    
    st.markdown("""
    Select allocations for three key assets below. The platform will compute historical correlation structures, 
    expected annual returns, annualized portfolio standard deviation (volatility), and output the 
    **Sharpe Ratio** to demonstrate mathematical financial modeling.
    """)

    # Let user select 3 tickers
    tickers_list = list(STOCK_PROFILES.keys())
    
    st.markdown("### 1. Select Portfolio Assets")
    col_p1, col_p2, col_p3 = st.columns(3)
    p_tick1 = col_p1.selectbox("Asset 1:", tickers_list, index=0)
    p_tick2 = col_p2.selectbox("Asset 2:", tickers_list, index=1)
    p_tick3 = col_p3.selectbox("Asset 3:", tickers_list, index=2)

    st.markdown("### 2. Set Asset Allocation Weights (Must sum to 100%)")
    col_w1, col_w2, col_w3 = st.columns(3)
    w1 = col_w1.slider(f"{p_tick1} Weight (%):", min_value=0, max_value=100, value=40, step=5)
    w2 = col_w2.slider(f"{p_tick2} Weight (%):", min_value=0, max_value=100, value=35, step=5)
    w3 = col_w3.slider(f"{p_tick3} Weight (%):", min_value=0, max_value=100, value=25, step=5)

    total_weight = w1 + w2 + w3
    st.markdown(f"**Total Portfolio Weight:** `{total_weight}%`")
    
    if total_weight != 100:
        st.error("❌ The sum of portfolio weights must equal exactly 100% to calculate metrics correctly.")
        st.stop()

    # Calculations
    try:
        # Load historical close prices for selected stocks from SQL
        weights = np.array([w1/100, w2/100, w3/100])
        
        # Load close prices in a single dataframe
        close_dfs = []
        for tick in [p_tick1, p_tick2, p_tick3]:
            df_tick = load_stock_data_from_db(tick)
            # Reindex by date, keep only close
            df_tick = df_tick[['date', 'close']].rename(columns={'close': tick})
            close_dfs.append(df_tick)
            
        # Merge
        merged_df = close_dfs[0]
        for df_sub in close_dfs[1:]:
            merged_df = pd.merge(merged_df, df_sub, on='date', how='inner')
            
        merged_df = merged_df.sort_values('date').set_index('date')
        
        # Calculate daily returns
        returns_df = merged_df.pct_change().dropna()
        
        # 1. Expected Returns (Annualized - assuming 252 trading days)
        mean_daily_returns = returns_df.mean()
        expected_annual_returns = np.dot(mean_daily_returns, weights) * 252
        
        # 2. Portfolio Volatility (Annualized)
        cov_matrix = returns_df.cov() * 252
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # 3. Sharpe Ratio (assuming Risk-Free Rate of 4%)
        rf_rate = 0.04
        sharpe_ratio = (expected_annual_returns - rf_rate) / portfolio_volatility

        st.success("✅ Portfolio Optimization Calculations Complete!")
        
        # Display Portfolio KPIs
        col_pkpi1, col_pkpi2, col_pkpi3 = st.columns(3)
        col_pkpi1.metric("Expected Annual Portfolio Return", f"{expected_annual_returns*100:.2f}%")
        col_pkpi2.metric("Expected Annual Portfolio Volatility", f"{portfolio_volatility*100:.2f}%")
        col_pkpi3.metric(
            "Sharpe Ratio (Risk-Adjusted Performance)", 
            f"{sharpe_ratio:.2f}",
            delta="Premium Returns" if sharpe_ratio > 1 else "Moderate Returns",
            delta_color="normal" if sharpe_ratio > 1 else "off"
        )
        
        st.markdown("---")
        
        # Visual breakdown of weights
        fig_pie = go.Figure(data=[go.Pie(labels=[p_tick1, p_tick2, p_tick3], values=[w1, w2, w3], hole=.4, marker=dict(colors=['#00f2fe', '#4facfe', '#9c27b0']))])
        fig_pie.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        
        col_pie, col_expl = st.columns([1, 2])
        with col_pie:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_expl:
            st.markdown("### 📊 Allocation Performance Summary")
            st.markdown(f"""
            - **Diversification Benefit**: The historical correlation between these assets helps reduce total portfolio volatility compared to holding a single high-beta asset like Tesla.
            - **Expected Annual Return ({expected_annual_returns*100:.2f}%)**: Represents the weighted average of historical annualized performance over our 5-year sample date scope.
            - **Portfolio Sharpe Ratio ({sharpe_ratio:.2f})**: Measures the excess return per unit of standard deviation. A Sharpe ratio **above 1.0** is considered excellent in traditional hedge fund strategies, denoting highly efficient risk-adjusted asset allocation.
            """)

    except Exception as e:
        st.error(f"Failed to compute portfolio statistics: {str(e)}")
