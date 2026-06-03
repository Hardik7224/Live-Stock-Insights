import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("Indicators")


def calculate_sma(df: pd.DataFrame, window: int, column: str = "Close") -> pd.Series:
    """
    Calculates the Simple Moving Average (SMA) for a given window.
    
    Formula:
    SMA_t = (P_t + P_{t-1} + ... + P_{t-N+1}) / N
    
    Business Significance:
    Filters out short-term price noise to help analysts identify primary trends.
    """
    if len(df) < window:
        logger.warning(f"Dataframe length ({len(df)}) is less than SMA window ({window}). Returns will contain NaN.")
    return df[column].rolling(window=window).mean()


def calculate_ema(df: pd.DataFrame, window: int, column: str = "Close") -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA) for a given window.
    
    Formula:
    EMA_t = P_t * alpha + EMA_{t-1} * (1 - alpha)
    where alpha = 2 / (window + 1)
    
    Business Significance:
    Unlike SMA, EMA gives more weight to recent prices, making it highly responsive 
    to recent market changes and trend breakouts.
    """
    if len(df) < window:
        logger.warning(f"Dataframe length ({len(df)}) is less than EMA window ({window}). Returns will contain NaN.")
    return df[column].ewm(span=window, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, window: int = 14, column: str = "Close") -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI), a popular momentum oscillator.
    
    Formula:
    RSI = 100 - (100 / (1 + RS))
    where RS = Smooth Average Gain / Smooth Average Loss
    
    Interpretation:
    RSI ranges from 0 to 100. 
    - > 70 indicates an overbought condition (asset is potentially overvalued, candidate for short/sale).
    - < 30 indicates an oversold condition (asset is potentially undervalued, candidate for buy/long).
    """
    if len(df) < window:
        logger.warning(f"Dataframe length ({len(df)}) is less than RSI window ({window}). Returns will contain NaN.")
        return pd.Series(index=df.index, dtype='float64')

    delta = df[column].diff()
    
    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    # Calculate initial wilder exponential averages
    # Wilder's smoothing technique:
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    
    # Apply exponential smoothing for subsequent periods
    # for t > window, avg_gain_t = (avg_gain_{t-1} * 13 + gain_t) / 14
    # We can approximate this using ewm with alpha = 1 / window
    avg_gain_smoothed = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss_smoothed = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    
    rs = avg_gain_smoothed / avg_loss_smoothed.replace(0, np.nan) # Prevent division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Handle zero division edge cases
    rsi = rsi.fillna(50) # Fallback to neutral 50 if both gain and loss are 0
    return rsi


def calculate_macd(
    df: pd.DataFrame, 
    fast_window: int = 12, 
    slow_window: int = 26, 
    signal_window: int = 9, 
    column: str = "Close"
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculates the Moving Average Convergence Divergence (MACD).
    
    Formulas:
    MACD Line = EMA(Close, 12) - EMA(Close, 26)
    Signal Line = EMA(MACD Line, 9)
    MACD Histogram = MACD Line - Signal Line
    
    Business Significance:
    A powerful trend-following momentum indicator. Crossovers between the MACD Line
    and the Signal Line are widely used as buy/sell indicators in trading systems.
    """
    if len(df) < slow_window:
        logger.warning(f"Dataframe length ({len(df)}) is less than slow MACD window ({slow_window}).")
        
    fast_ema = df[column].ewm(span=fast_window, adjust=False).mean()
    slow_ema = df[column].ewm(span=slow_window, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram


def calculate_bollinger_bands(
    df: pd.DataFrame, 
    window: int = 20, 
    num_std: float = 2.0, 
    column: str = "Close"
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculates the Bollinger Bands (Upper, Lower, Middle, and Bandwidth).
    
    Formulas:
    Middle Band = 20 SMA
    Upper Band = 20 SMA + (2 * 20 Rolling Standard Deviation)
    Lower Band = 20 SMA - (2 * 20 Rolling Standard Deviation)
    Bandwidth = (Upper Band - Lower Band) / Middle Band
    
    Business Significance:
    Indicates high/low volatility. Prices tend to stay within bands. Touch or penetration of bands:
    - Upper Band: Asset is overextended (potential resistance, overbought).
    - Lower Band: Asset is depressed (potential support, oversold).
    """
    if len(df) < window:
        logger.warning(f"Dataframe length ({len(df)}) is less than Bollinger window ({window}).")
        
    middle_band = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()
    
    upper_band = middle_band + (num_std * rolling_std)
    lower_band = middle_band - (num_std * rolling_std)
    
    bandwidth = (upper_band - lower_band) / middle_band.replace(0, np.nan)
    
    return upper_band, middle_band, lower_band, bandwidth


def enrich_with_indicators(df: pd.DataFrame, column: str = "Close") -> pd.DataFrame:
    """
    Helper function to compute and add all core technical indicators to a DataFrame.
    """
    df_copy = df.copy()
    
    # 1. SMAs
    df_copy["SMA_20"] = calculate_sma(df_copy, window=20, column=column)
    df_copy["SMA_50"] = calculate_sma(df_copy, window=50, column=column)
    
    # 2. EMA
    df_copy["EMA_20"] = calculate_ema(df_copy, window=20, column=column)
    
    # 3. RSI
    df_copy["RSI_14"] = calculate_rsi(df_copy, window=14, column=column)
    
    # 4. MACD
    macd, signal, hist = calculate_macd(df_copy, fast_window=12, slow_window=26, signal_window=9, column=column)
    df_copy["MACD_Line"] = macd
    df_copy["MACD_Signal"] = signal
    df_copy["MACD_Hist"] = hist
    
    # 5. Bollinger Bands
    upper, middle, lower, bandwidth = calculate_bollinger_bands(df_copy, window=20, num_std=2.0, column=column)
    df_copy["BB_Upper"] = upper
    df_copy["BB_Middle"] = middle
    df_copy["BB_Lower"] = lower
    df_copy["BB_Bandwidth"] = bandwidth
    
    return df_copy
