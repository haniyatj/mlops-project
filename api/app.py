"""
Flask API for serving the linear regression stock prediction model.
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load model on startup
MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    '../models/linear_regression_model.joblib')
logger.info(f"Loading model from {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None


def compute_rsi(price_series, period=14):
    """Compute Relative Strength Index."""
    delta = price_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Handle division by zero
    avg_loss = avg_loss.replace(0, np.finfo(float).eps)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def prepare_data_from_json(data_json, target_col='close', lookback=5):
    """Prepare data from JSON input for prediction."""
    # Convert JSON to DataFrame
    df = pd.DataFrame(data_json)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    # Create features similar to training
    df_copy = df.copy()

    # Add technical indicators
    df_copy['MA5'] = df_copy[target_col].rolling(window=5).mean()
    df_copy['MA20'] = df_copy[target_col].rolling(window=20).mean()
    df_copy['RSI'] = compute_rsi(df_copy[target_col])

    # Create lagged features
    for i in range(1, lookback + 1):
        df_copy[f'{target_col}_lag_{i}'] = df_copy[target_col].shift(i)

    # Add volatility and momentum features
    df_copy['volatility'] = df_copy[target_col].rolling(window=10).std()
    df_copy['momentum'] = df_copy[target_col].pct_change(periods=5)

    # Remove rows with NaN values
    df_copy.dropna(inplace=True)

    # Extract features
    features = df_copy.drop(columns=[target_col])

    return features, df_copy[target_col]


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if model is not None:
        return jsonify({
            "status": "healthy",
            "model": "linear_regression",
            "model_loaded": True
        })
    else:
        return jsonify({
            "status": "unhealthy",
            "model": "linear_regression",
            "model_loaded": False
        }), 503


@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the linear regression model."""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded"}), 503

        # Parse request data
        request_data = request.get_json()

        if not request_data or 'data' not in request_data:
            return jsonify(
                {"error": "No data provided or invalid format"}), 400

        # Get stock data from request
        stock_data = request_data['data']

        # Prepare data for prediction
        features, actual = prepare_data_from_json(
            stock_data, target_col='close', lookback=5)

        # Make predictions
        predictions = model.predict(features)

        # Format for response
        data = []
        for i, (idx, act, pred) in enumerate(
                zip(actual.index, actual, predictions)):
            data.append({
                "date": (
                    idx.strftime('%Y-%m-%d')
                    if hasattr(idx, 'strftime')
                    else str(idx)
                ),
                "actual": float(act),
                "predicted": float(pred)
            })

        # Calculate error metrics
        mse = (
            (np.array([item["actual"] for item in data])
            - np.array([item["predicted"] for item in data])) ** 2
            ).mean()
        rmse = np.sqrt(mse)
        mae = (
            np.abs(np.array([item["actual"] for item in data])
            - np.array([item["predicted"] for item in data]))
        ).mean()

        return jsonify({
            "model": "linear_regression",
            "predictions": data,
            "metrics": {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae)
            }
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
