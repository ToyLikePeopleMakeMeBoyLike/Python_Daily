#!/usr/bin/env python3

"""
Time Series Forecasting of Stock Prices Using Pandas and TensorFlow (Keras)

This script demonstrates how to:
1. Download stock price data using yfinance.
2. Use pandas to clean and manipulate the dataset.
3. Perform time series feature engineering (windowing).
4. Build and train an LSTM model in TensorFlow.
5. Evaluate and visualize the results.

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(64, input_shape=(60, 1)))




# ------------------------------------------------------------------------------
# 1. Configuration and Hyperparameters
# ------------------------------------------------------------------------------
STOCK_SYMBOL = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2022-12-31"
WINDOW_SIZE = 60         # number of days to look back for each training sample
TEST_SPLIT_RATIO = 0.2   # 20% of data for testing
BATCH_SIZE = 32
EPOCHS = 10
UNITS = 64               # LSTM units
DROPOUT_RATE = 0.2       # dropout rate


# ------------------------------------------------------------------------------
# 2. Data Loading
# ------------------------------------------------------------------------------
def load_stock_data(symbol, start, end):
    """
    Downloads stock data for a given symbol between start and end dates.

    :param symbol: Stock symbol to download (e.g. "AAPL")
    :param start: Start date as string (e.g. "2015-01-01")
    :param end: End date as string (e.g. "2022-12-31")
    :return: pandas DataFrame with the downloaded data
    """
    df = yf.download(symbol, start=start, end=end)
    return df


# ------------------------------------------------------------------------------
# 3. Data Preprocessing
# ------------------------------------------------------------------------------
def preprocess_data(df):
    """
    Preprocesses the stock data:
    - Select the 'Close' price column for modeling.
    - Scales the data to [0, 1] using MinMaxScaler.

    :param df: DataFrame containing stock data with at least 'Close' column
    :return: scaled_data (np.array), scaler (MinMaxScaler)
    """
    # We only use the 'Close' column for forecasting
    close_prices = df[['Close']].values
    
    # Scale data to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    return scaled_data, scaler


def create_sequences(dataset, window_size):
    """
    Creates sequences (X, y) pairs from a time-series dataset.

    :param dataset: Scaled time-series data (numpy array)
    :param window_size: Number of time steps to include in each sample
    :return: X, y (numpy arrays)
    """
    X = []
    y = []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i-window_size:i, 0])  # from i-window_size to i-1
        y.append(dataset[i, 0])               # the i-th value is the target
    return np.array(X), np.array(y)


def split_train_test(X, y, test_ratio):
    """
    Splits the dataset into training and testing.

    :param X: Features array
    :param y: Targets array
    :param test_ratio: Fraction of data to be used for testing
    :return: X_train, y_train, X_test, y_test
    """
    split_index = int(len(X) * (1 - test_ratio))
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]
    return X_train, y_train, X_test, y_test


# ------------------------------------------------------------------------------
# 4. Model Definition
# ------------------------------------------------------------------------------
def build_lstm_model(window_size, units=64, dropout_rate=0.2):
    """
    Builds and compiles an LSTM model using TensorFlow Keras.

    :param window_size: Number of time steps in each input sequence
    :param units: Number of LSTM units
    :param dropout_rate: Dropout rate for regularization
    :return: Compiled Keras model
    """
    model = Sequential()
    # LSTM layer
    model.add(LSTM(units=units, return_sequences=False, 
                   input_shape=(window_size, 1)))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model


# ------------------------------------------------------------------------------
# 5. Main Execution
# ------------------------------------------------------------------------------
def main():
    print("[INFO] Downloading data...")
    df = load_stock_data(STOCK_SYMBOL, START_DATE, END_DATE)
    
    # Display some basic information
    print(df.head())
    print(df.info())

    print("\n[INFO] Preprocessing data...")
    scaled_data, scaler = preprocess_data(df)

    print("[INFO] Creating sequences for training...")
    X, y = create_sequences(scaled_data, WINDOW_SIZE)

    # Reshape X to fit LSTM input shape: (samples, time steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    print("[INFO] Splitting into train and test sets...")
    X_train, y_train, X_test, y_test = split_train_test(X, y, TEST_SPLIT_RATIO)

    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Test shape: {X_test.shape}, {y_test.shape}")

    print("\n[INFO] Building LSTM model...")
    model = build_lstm_model(WINDOW_SIZE, UNITS, DROPOUT_RATE)
    model.summary()

    print("[INFO] Training the model...")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1
    )

    print("[INFO] Evaluating the model...")
    # Evaluate on the test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss}")

    print("\n[INFO] Generating predictions...")
    predictions = model.predict(X_test)
    # Invert scaling
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # ------------------------------------------------------------------------------
    # 6. Visualization
    # ------------------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled, label="True")
    plt.plot(predictions_rescaled, label="Predicted")
    plt.title(f"{STOCK_SYMBOL} Stock Price Prediction")
    plt.xlabel("Time (Days in Test Set)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

    # Optional: Save the model
    print("[INFO] Saving the model...")
    model.save("lstm_stock_model.keras")
    print("[INFO] Model saved to 'lstm_stock_model.keras'.")


if __name__ == "__main__":
    main()
