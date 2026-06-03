import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logger = logging.getLogger("MachineLearningPipeline")


class StockMLPipeline:
    """
    An enterprise-grade ML pipeline that reads preprocessed stock price data,
    creates temporal features, trains multiple regression models, evaluates performance,
    and serializes the best performing models for deployment in the dashboard.
    """

    def __init__(
        self, 
        processed_data_dir: str = "data/processed", 
        model_save_dir: str = "models/saved_models"
    ):
        self.processed_data_dir = processed_data_dir
        self.model_save_dir = model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)
        logger.info(f"Initialized StockMLPipeline. Models will be saved to: {self.model_save_dir}")

    def prepare_time_series_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Creates historical lag features and engineers the target variable (Next Day's Close).
        
        Args:
            df (pd.DataFrame): Preprocessed stock DataFrame.
            
        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame]: 
                - X: Feature matrix for training (historical records).
                - y: Target series for training.
                - X_latest: Feature vector representing the most recent day (used for forward forecasting).
        """
        df_ml = df.copy()
        
        # Enforce chronological ordering
        date_col = 'Date' if 'Date' in df_ml.columns else 'Datetime'
        df_ml[date_col] = pd.to_datetime(df_ml[date_col])
        df_ml = df_ml.sort_values(date_col).reset_index(drop=True)

        # 1. Create Lag Features (highly critical for auto-regressive structures)
        for lag in [1, 2, 3, 5]:
            df_ml[f'Close_Lag_{lag}'] = df_ml['Close'].shift(lag)
            df_ml[f'Volume_Lag_{lag}'] = df_ml['Volume'].shift(lag)
            df_ml[f'Return_Lag_{lag}'] = df_ml['Daily_Return'].shift(lag)

        # 2. Target Variable: Next Day's Close (shifted -1)
        df_ml['Target_Close'] = df_ml['Close'].shift(-1)

        # 3. Features list to use for ML models
        # We select technical indicators and lag variables, excluding metadata or leakage fields
        feature_cols = [
            'Close', 'Volume', 'Daily_Return', 'Volatility_20d',
            'SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 
            'MACD_Line', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Bandwidth',
            'Close_Lag_1', 'Volume_Lag_1', 'Return_Lag_1',
            'Close_Lag_2', 'Volume_Lag_2', 'Return_Lag_2',
            'Close_Lag_3', 'Volume_Lag_3', 'Return_Lag_3',
            'Close_Lag_5', 'Volume_Lag_5', 'Return_Lag_5'
        ]

        # Ensure all columns exist before selecting
        selected_features = [col for col in feature_cols if col in df_ml.columns]

        # Isolate the final row where 'Target_Close' is NaN.
        # This row contains the features of "TODAY" to predict "TOMORROW'S CLOSE"
        latest_row = df_ml.tail(1)
        X_latest = latest_row[selected_features]

        # For historical training, we drop rows with NaNs (which includes the lags at the start 
        # and the Target_Close shift at the end)
        df_train = df_ml.dropna(subset=selected_features + ['Target_Close'])

        X = df_train[selected_features]
        y = df_train['Target_Close']

        return X, y, X_latest

    def train_and_evaluate(self, ticker: str) -> Dict[str, Any]:
        """
        Loads preprocessed data for a ticker, splits chronologically, trains multiple models,
        evaluates metrics, and serializes the best performing model.
        """
        logger.info(f"=== Beginning ML Training Pipeline for Ticker: {ticker} ===")
        
        # Load preprocessed historical file
        file_name = f"{ticker.replace('.', '_')}_historical_processed.csv"
        file_path = os.path.join(self.processed_data_dir, file_name)
        
        if not os.path.exists(file_path):
            logger.error(f"Processed CSV not found for {ticker} at {file_path}")
            return {}

        df = pd.read_csv(file_path)
        X, y, X_latest = self.prepare_time_series_features(df)

        if len(X) < 100:
            logger.warning(f"Insufficient training samples ({len(X)}) for ticker {ticker}. Skipping.")
            return {}

        # Chronological Split (Train on 80% historical, Test on 20% future)
        # Prevents temporal leakage compared to random train_test_split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Split data: Train samples = {len(X_train)}, Test samples = {len(X_test)}")

        models = {
            "Linear_Regression": LinearRegression(),
            "Random_Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient_Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.05)
        }

        model_results = {}
        best_r2 = -float("inf")
        best_model_name = ""
        best_model_obj = None

        for name, model in models.items():
            logger.info(f"Training model: {name}...")
            model.fit(X_train, y_train)
            
            # Predict
            preds_test = model.predict(X_test)
            preds_train = model.predict(X_train)

            # Evaluate Metrics
            mae = mean_absolute_error(y_test, preds_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds_test))
            r2 = r2_score(y_test, preds_test)

            mae_train = mean_absolute_error(y_train, preds_train)
            r2_train = r2_score(y_train, preds_train)

            logger.info(f"[{name}] Test Metrics -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
            logger.info(f"[{name}] Train Metrics -> MAE: {mae_train:.2f}, R2: {r2_train:.4f}")

            model_results[name] = {
                "model": model,
                "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
                "train_metrics": {"MAE": mae_train, "R2": r2_train}
            }

            # Select Best Model based on R2 score on validation/test set
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_model_obj = model

        # Check for extreme overfitting (e.g. high train R2 but negative test R2)
        # Fallback to Linear Regression if ensemble models overfit severely
        if best_r2 < 0:
            logger.warning(f"All models underperformed (negative R2) on test set for {ticker}. Selecting Linear Regression as robust fallback.")
            best_model_name = "Linear_Regression"
            best_model_obj = models["Linear_Regression"]
            best_r2 = model_results["Linear_Regression"]["metrics"]["R2"]

        logger.info(f"*** Best Model for {ticker} is: {best_model_name} with Test R2 = {best_r2:.4f} ***")

        # -------------------------------------------------------------------------
        # SERIALIZE MODELS
        # -------------------------------------------------------------------------
        # Save Ticker-Specific Best Model
        best_model_path = os.path.join(self.model_save_dir, f"{ticker.replace('.', '_')}_best_model.pkl")
        joblib.dump(best_model_obj, best_model_path)
        
        # Save X_latest vector to easily load in Streamlit for inference
        latest_vector_path = os.path.join(self.model_save_dir, f"{ticker.replace('.', '_')}_latest_vector.pkl")
        joblib.dump(X_latest, latest_vector_path)

        # Save feature column names to ensure input alignment during prediction
        features_path = os.path.join(self.model_save_dir, f"{ticker.replace('.', '_')}_feature_cols.pkl")
        joblib.dump(list(X.columns), features_path)

        logger.info(f"Serialized serialized assets for {ticker} to {self.model_save_dir}")

        return {
            "ticker": ticker,
            "best_model_name": best_model_name,
            "metrics": model_results[best_model_name]["metrics"],
            "features_count": len(X.columns)
        }

    def train_all_tickers(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Runs the ML pipeline for all specified stock symbols sequentially.
        """
        summary = {}
        for ticker in tickers:
            try:
                results = self.train_and_evaluate(ticker)
                if results:
                    summary[ticker] = results
            except Exception as e:
                logger.error(f"ML Pipeline failed for ticker {ticker}: {str(e)}")
        
        logger.info("\n=========================================")
        logger.info("MACHINE LEARNING MULTI-STOCK RUN SUMMARY:")
        logger.info("=========================================")
        for ticker, info in summary.items():
            logger.info(
                f"- {ticker}: Best Model = {info['best_model_name']} | "
                f"Test R2 = {info['metrics']['R2']:.4f} | "
                f"Test RMSE = {info['metrics']['RMSE']:.2f}"
            )
        return summary


if __name__ == "__main__":
    # Test execution
    TICKERS = ["AAPL", "TSLA", "MSFT", "RELIANCE.NS", "TCS.NS"]
    pipeline = StockMLPipeline(processed_data_dir="data/processed", model_save_dir="models/saved_models")
    pipeline.train_all_tickers(TICKERS)
