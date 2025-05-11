import mlflow.keras
import mlflow.sklearn
import mlflow
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables oneDNN optimizations

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")


def load_data(file_path):
    """Load and prepare stock data from CSV file."""
    df = pd.read_csv(file_path)
    # Check if Date column exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    # Print column names for debugging
    print(f"Available columns in the dataset: {df.columns.tolist()}")
    return df


def create_features(df, target_col='close', window_size=5):
    """Create features for prediction based on historical window."""
    df_copy = df.copy()

    print(f"Target column: {target_col}")

    # Add technical indicators
    df_copy['MA5'] = df_copy[target_col].rolling(window=5).mean()
    df_copy['MA20'] = df_copy[target_col].rolling(window=20).mean()
    df_copy['RSI'] = compute_rsi(df_copy[target_col])

    # Create lagged features
    for i in range(1, window_size + 1):
        df_copy[f'{target_col}_lag_{i}'] = df_copy[target_col].shift(i)

    # Add volatility features
    df_copy['volatility'] = df_copy[target_col].rolling(window=10).std()

    # Add price momentum
    df_copy['momentum'] = df_copy[target_col].pct_change(periods=5)

    # Remove rows with NaN values
    df_copy.dropna(inplace=True)

    # Extract features and target
    features = df_copy.drop(columns=[target_col])
    target = df_copy[target_col]

    return features, target


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


def prepare_data_for_lstm(features, target, lookback=5):
    """Prepare data for LSTM model by reshaping into time sequences."""
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features.iloc[i:i + lookback].values)
        y.append(target.iloc[i + lookback])
    return np.array(X), np.array(y)


def train_linear_regression(
        X_train,
        X_test,
        y_train,
        y_test,
        experiment_name="stock_prediction"):
    """Train and evaluate a linear regression model with MLflow tracking."""
    with mlflow.start_run(run_name="linear_regression") as run:
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Log parameters
        mlflow.log_param("model_type", "Linear Regression")

        # Log metrics
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)

        # Log model
        mlflow.sklearn.log_model(model, "linear_regression_model")

        # Plot predictions
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.values, label='Actual')
        plt.plot(y_test_pred, label='Predicted')
        plt.title('Linear Regression: Actual vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()

        # Save and log the plot
        plot_path = "lr_predictions.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)  # Clean up

        print(f"Linear Regression Test RMSE: {test_rmse:.4f}")
        return model, run.info.run_id


def train_random_forest(
        X_train,
        X_test,
        y_train,
        y_test,
        experiment_name="stock_prediction"):
    """Train and evaluate a Random Forest model with MLflow tracking."""
    with mlflow.start_run(run_name="random_forest") as run:
        # Train the model with hyperparameters
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Log parameters
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("min_samples_split", 5)
        mlflow.log_param("min_samples_leaf", 2)

        # Log metrics
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)

        # Log feature importances
        feature_importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # Create feature importance plot
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importances['Feature'][:10],
                 feature_importances['Importance'][:10])
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()

        # Save and log the feature importance plot
        fi_plot_path = "rf_feature_importance.png"
        plt.savefig(fi_plot_path)
        mlflow.log_artifact(fi_plot_path)
        os.remove(fi_plot_path)  # Clean up

        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Plot predictions
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.values, label='Actual')
        plt.plot(y_test_pred, label='Predicted')
        plt.title('Random Forest: Actual vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()

        # Save and log the plot
        plot_path = "rf_predictions.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)  # Clean up

        print(f"Random Forest Test RMSE: {test_rmse:.4f}")
        return model, run.info.run_id


def train_lstm_model(X_train, X_test, y_train, y_test,
                     experiment_name="stock_prediction"):
    """Train and evaluate an LSTM model with MLflow tracking."""
    with mlflow.start_run(run_name="lstm_model") as run:
        # Define LSTM model architecture
        model = Sequential([LSTM(units=50,
                                 return_sequences=True,
                                 input_shape=(X_train.shape[1],
                                              X_train.shape[2])),
                            Dropout(0.2),
                            LSTM(units=50),
                            Dropout(0.2),
                            Dense(units=1)])

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Log model parameters
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("lstm_units", 50)
        mlflow.log_param("dropout_rate", 0.2)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("loss_function", "mean_squared_error")

        # Train the model with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            epochs=50,  # Reduced from 100 for faster execution
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred.flatten())

        # Log metrics
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)

        # Log the model
        mlflow.keras.log_model(model, "lstm_model")

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('LSTM Model: Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        history_plot_path = "lstm_training_history.png"
        plt.savefig(history_plot_path)
        mlflow.log_artifact(history_plot_path)
        os.remove(history_plot_path)  # Clean up

        # Plot predictions
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual')
        plt.plot(y_test_pred, label='Predicted')
        plt.title('LSTM: Actual vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()

        pred_plot_path = "lstm_predictions.png"
        plt.savefig(pred_plot_path)
        mlflow.log_artifact(pred_plot_path)
        os.remove(pred_plot_path)  # Clean up

        print(f"LSTM Test RMSE: {test_rmse:.4f}")
        return model, run.info.run_id

# Modify the main() function to save the best model


def main():
    # Set MLflow experiment
    mlflow.set_experiment("stock_prediction")

    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Load data
    data_path = os.path.join('data', 'AAPL_daily_cleaned.csv')
    print(f"Loading data from: {data_path}")

    try:
        stock_data = load_data(data_path)

        # Create features - explicitly passing 'close' as target column
        features, target = create_features(stock_data, target_col='close')

        # Train/test split for traditional ML
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, shuffle=False
        )

        # Train Linear Regression model
        lr_model, lr_run_id = train_linear_regression(
            X_train, X_test, y_train, y_test)

        # Train Random Forest model
        rf_model, rf_run_id = train_random_forest(
            X_train, X_test, y_train, y_test)

        # Prepare data for LSTM
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(features)
        y_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

        # Convert to DataFrame to maintain time indexing
        X_scaled_df = pd.DataFrame(
            X_scaled,
            index=features.index,
            columns=features.columns)
        y_scaled_df = pd.DataFrame(
            y_scaled, index=target.index, columns=['close'])

        # Train/test split for LSTM
        train_size = int(len(X_scaled_df) * 0.8)
        X_train_scaled = X_scaled_df.iloc[:train_size]
        X_test_scaled = X_scaled_df.iloc[train_size:]
        y_train_scaled = y_scaled_df.iloc[:train_size]
        y_test_scaled = y_scaled_df.iloc[train_size:]

        # Prepare sequences for LSTM
        lookback = 5
        X_train_lstm, y_train_lstm = prepare_data_for_lstm(
            X_train_scaled, y_train_scaled, lookback)
        X_test_lstm, y_test_lstm = prepare_data_for_lstm(
            X_test_scaled, y_test_scaled, lookback)

        # Reshape for LSTM [samples, time steps, features]
        X_train_lstm = X_train_lstm.reshape(
            (X_train_lstm.shape[0], lookback, X_train_scaled.shape[1]))
        X_test_lstm = X_test_lstm.reshape(
            (X_test_lstm.shape[0], lookback, X_test_scaled.shape[1]))

        # Train LSTM model
        lstm_model, lstm_run_id = train_lstm_model(
            X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm)

        print(f"Linear Regression Run ID: {lr_run_id}")
        print(f"Random Forest Run ID: {rf_run_id}")
        print(f"LSTM Run ID: {lstm_run_id}")

        # Register the best model (based on metrics)
        client = mlflow.tracking.MlflowClient()

        # Get run metrics
        lr_metrics = client.get_run(lr_run_id).data.metrics
        rf_metrics = client.get_run(rf_run_id).data.metrics
        lstm_metrics = client.get_run(lstm_run_id).data.metrics

        # Choose the best model based on test RMSE
        best_rmse = float('inf')
        best_model = None
        best_model_type = None
        best_scalers = None

        if lr_metrics['test_rmse'] < best_rmse:
            best_rmse = lr_metrics['test_rmse']
            best_model = lr_model
            best_model_type = 'linear_regression'
            best_model_name = 'Linear Regression'

        if rf_metrics['test_rmse'] < best_rmse:
            best_rmse = rf_metrics['test_rmse']
            best_model = rf_model
            best_model_type = 'random_forest'
            best_model_name = 'Random Forest'

        if lstm_metrics['test_rmse'] < best_rmse:
            best_rmse = lstm_metrics['test_rmse']
            best_model = lstm_model
            best_model_type = 'lstm'
            best_model_name = 'LSTM'
            best_scalers = (scaler_X, scaler_y)

        # Save the best model to disk
        if best_model_type in ['linear_regression', 'random_forest']:
            model_path = models_dir / f"{best_model_type}_model.joblib"
            joblib.dump(best_model, model_path)
            print(f"Saved {best_model_name} model to {model_path}")
        elif best_model_type == 'lstm':
            # Save Keras model
            model_path = models_dir / "lstm_model.keras"
            best_model.save(model_path)

            # Save scalers
            scaler_X_path = models_dir / "lstm_scaler_X.joblib"
            scaler_y_path = models_dir / "lstm_scaler_y.joblib"
            joblib.dump(best_scalers[0], scaler_X_path)
            joblib.dump(best_scalers[1], scaler_y_path)
            print(
                f"Saved LSTM model to {model_path} and scalers to {scaler_X_path}, {scaler_y_path}")

        # Register the best model in MLflow Model Registry
        model_name, run_id, artifact_path, registry_name = best_model_name, None, None, None

        if best_model_type == 'linear_regression':
            run_id = lr_run_id
            artifact_path = 'linear_regression_model'
            registry_name = 'stock_prediction_linear_regression'
        elif best_model_type == 'random_forest':
            run_id = rf_run_id
            artifact_path = 'random_forest_model'
            registry_name = 'stock_prediction_random_forest'
        elif best_model_type == 'lstm':
            run_id = lstm_run_id
            artifact_path = 'lstm_model'
            registry_name = 'stock_prediction_lstm'

        print(
            f"{model_name} model performed best with RMSE: {best_rmse:.4f}. Registering this model...")

        model_path = f"runs:/{run_id}/{artifact_path}"
        model_version = mlflow.register_model(model_path, registry_name)
        print(f"Registered {registry_name} as version {model_version.version}")

        # Create comparison plot of all models
        plt.figure(figsize=(12, 8))
        plt.bar(['Linear Regression', 'Random Forest', 'LSTM'], [
                lr_metrics['test_rmse'], rf_metrics['test_rmse'], lstm_metrics['test_rmse']])
        plt.title('Model Comparison - Test RMSE')
        plt.ylabel('RMSE (lower is better)')
        plt.ylim(bottom=0)

        comparison_plot_path = "model_comparison.png"
        plt.savefig(comparison_plot_path)

        # Log comparison plot to all runs
        with mlflow.start_run(run_id=lr_run_id):
            mlflow.log_artifact(comparison_plot_path)
        with mlflow.start_run(run_id=rf_run_id):
            mlflow.log_artifact(comparison_plot_path)
        with mlflow.start_run(run_id=lstm_run_id):
            mlflow.log_artifact(comparison_plot_path)

        os.remove(comparison_plot_path)  # Clean up

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
