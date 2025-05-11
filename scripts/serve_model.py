"""
Serve saved models for making stock predictions.

This script loads a saved model from the models directory and uses it to make predictions.
"""

import os
import pandas as pd
import numpy as np
import joblib
import argparse
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


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


def prepare_data_for_serving(file_path, target_col='close', lookback=5):
    """Prepare data for prediction when serving the model."""
    # Load and preprocess data
    df = pd.read_csv(file_path)
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

    # Add volatility features that were missing
    df_copy['volatility'] = df_copy[target_col].rolling(window=10).std()

    # Add price momentum that was missing
    df_copy['momentum'] = df_copy[target_col].pct_change(periods=5)

    # Remove rows with NaN values
    df_copy.dropna(inplace=True)

    # Extract features
    features = df_copy.drop(columns=[target_col])

    return features, df_copy[target_col]


def prepare_data_for_lstm(features, target, lookback=5):
    """Prepare data for LSTM model by reshaping into time sequences."""
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features.iloc[i:i + lookback].values)
        y.append(target.iloc[i + lookback])
    return np.array(X), np.array(y)


def load_saved_model(model_type):
    """Load the locally saved model from the models directory."""
    # Get the absolute path to the models directory (one level up from scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    models_dir = os.path.join(project_root, "models")

    print(f"Looking for models in: {models_dir}")  # Debug output

    if model_type == "lstm":
        model_path = os.path.join(models_dir, "lstm_model.keras")
        scaler_X_path = os.path.join(models_dir, "lstm_scaler_X.joblib")
        scaler_y_path = os.path.join(models_dir, "lstm_scaler_y.joblib")

        print(f"LSTM model path: {model_path}")  # Debug output
        # Debug output
        print(f"Scaler paths: {scaler_X_path}, {scaler_y_path}")

        model = load_model(model_path)
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        return model, scaler_X, scaler_y
    else:
        model_path = os.path.join(models_dir, f"{model_type}_model.joblib")
        print(f"{model_type} model path: {model_path}")  # Debug output
        model = joblib.load(model_path)
        return model, None, None


def serve_saved_model(model_type, data_path):
    """Serve the specified model for predictions using locally saved models."""
    try:
        # Load the model and scalers (if LSTM)
        model, scaler_X, scaler_y = load_saved_model(model_type)
        is_lstm = model_type == "lstm"

        # Prepare data - using 'close' as target column
        features, actual = prepare_data_for_serving(
            data_path, target_col='close', lookback=5)

        # Make predictions based on model type
        if is_lstm:
            # For LSTM models we need to reshape the data
            X_scaled = scaler_X.transform(features)
            X_scaled_df = pd.DataFrame(
                X_scaled,
                index=features.index,
                columns=features.columns)

            # Prepare sequences for LSTM
            lookback = 5
            X_lstm, y_lstm = prepare_data_for_lstm(
                X_scaled_df, actual, lookback)

            # Reshape for LSTM [samples, time steps, features]
            X_lstm = X_lstm.reshape(
                (X_lstm.shape[0], lookback, X_scaled_df.shape[1]))

            # Make predictions
            predictions_scaled = model.predict(X_lstm)
            predictions = scaler_y.inverse_transform(predictions_scaled)

            # Adjust the results to match the actual values length
            # LSTM predictions will be shorter due to the lookback window
            actual = actual.iloc[lookback:].iloc[:len(predictions)]
        else:
            # Standard models (Linear Regression, Random Forest)
            predictions = model.predict(features)

        # Display results
        results = pd.DataFrame({
            'Actual': actual,
            'Predicted': predictions.flatten() if is_lstm else predictions
        })

        print("\nPrediction Results:")
        print(results.tail(10))

        # Calculate error metrics
        mse = ((results['Actual'] - results['Predicted']) ** 2).mean()
        rmse = np.sqrt(mse)
        mae = (results['Actual'] - results['Predicted']).abs().mean()

        print("\nError Metrics on Test Data:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        # Save predictions to CSV
        output_file = f"predictions_{model_type}.csv"
        results.to_csv(output_file)
        print(f"\nSaved predictions to {output_file}")

    except Exception as e:
        print(f"Error serving model: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Make sure the model files exist in the models directory.")


def main():
    parser = argparse.ArgumentParser(
        description="Serve locally saved model for predictions")
    parser.add_argument(
        "--model",
        default="linear_regression",
        help="Model type (linear_regression, random_forest, or lstm)")
    parser.add_argument(
        "--data",
        default="data/AAPL_daily_cleaned.csv",
        help="Path to input data")

    args = parser.parse_args()

    # Validate model type
    valid_models = ["linear_regression", "random_forest", "lstm"]
    if args.model not in valid_models:
        print(f"Invalid model type. Must be one of: {', '.join(valid_models)}")
        return

    serve_saved_model(args.model, args.data)


if __name__ == "__main__":
    main()
