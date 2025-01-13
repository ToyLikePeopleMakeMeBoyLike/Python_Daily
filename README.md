# LSTM Stock Price Forecasting: A Comprehensive Guide

This markdown file serves as a **step-by-step guide** for anyone wanting to use an LSTM-based model in TensorFlow/Keras for time series forecasting, with Pandas for data manipulation. It walks you through project setup, data loading, model building, training, evaluation, and visualization (including solutions for headless environments).

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Prerequisites](#prerequisites)  
3. [Project Structure](#project-structure)  
4. [Environment Setup](#environment-setup)  
5. [Script Walkthrough](#script-walkthrough)  
   - [Imports & Configuration](#imports--configuration)  
   - [Data Loading](#data-loading)  
   - [Data Preprocessing](#data-preprocessing)  
   - [Creating Sequences (Windowing)](#creating-sequences-windowing)  
   - [Splitting Train/Test](#splitting-traintest)  
   - [Model Building](#model-building)  
   - [Model Training & Evaluation](#model-training--evaluation)  
   - [Visualization](#visualization)  
   - [Model Saving](#model-saving)  
6. [Handling Visualization in Headless Environments](#handling-visualization-in-headless-environments)  
7. [Troubleshooting](#troubleshooting)  
8. [Extensions & Next Steps](#extensions--next-steps)  
9. [Conclusion](#conclusion)  

---

## Project Overview

This project demonstrates how to:

- **Download** historical stock data (e.g., AAPL) using `yfinance`  
- **Clean and preprocess** data with Pandas  
- **Scale** features using `MinMaxScaler`  
- **Create time windows** for LSTM training  
- **Build and train** an LSTM in TensorFlow/Keras  
- **Evaluate** the model performance on a test set  
- **Visualize** predictions vs. actual prices  
- **Save** the trained model for future use  

---

## Prerequisites

- A **working Python 3 environment** (3.7+ recommended for TensorFlow 2.x)  
- Knowledge of **virtual environments** or **conda** environments  
- Basic familiarity with **Pandas**, **NumPy**, **Matplotlib**, and **TensorFlow/Keras**  

---

## Project Structure

A suggested directory layout:

my_time_series_forecasting_project/ ├── README.md ├── requirements.txt └── stock_forecast.py

javascript
Copy code

- **`README.md`**: Provides an overview of the project (this file could serve that purpose).  
- **`requirements.txt`**: Contains a list of Python dependencies.  
- **`stock_forecast.py`**: Main script for downloading data, building the model, training, and visualization.

**Example** `requirements.txt`:
```txt
pandas
yfinance
numpy
scikit-learn
tensorflow
matplotlib
Environment Setup
Clone or download this repository (or copy the files into your workspace).
Create a virtual environment (optional, but recommended):
bash
Copy code
python -m venv venv
source venv/bin/activate  # Mac/Linux
# or venv\Scripts\activate on Windows
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Check your installation by verifying the versions:
bash
Copy code
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import yfinance as yf; print(yf.__version__)"
You should see no import errors and a valid version for TensorFlow (≥2.0).
Script Walkthrough
Below is an outline of what happens in stock_forecast.py.

Imports & Configuration
python
Copy code
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# User-defined parameters
STOCK_SYMBOL = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2022-12-31"
WINDOW_SIZE = 60
TEST_SPLIT_RATIO = 0.2
BATCH_SIZE = 32
EPOCHS = 10
UNITS = 64
DROPOUT_RATE = 0.2
Constants like STOCK_SYMBOL, START_DATE, and WINDOW_SIZE define how data is fetched and how the time series is windowed.
Hyperparameters such as EPOCHS and UNITS can be tweaked for better performance.
Data Loading
python
Copy code
def load_stock_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df
Uses yfinance to download the specified stock data (AAPL by default).
Returns a Pandas DataFrame.
Data Preprocessing
python
Copy code
def preprocess_data(df):
    close_prices = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    return scaled_data, scaler
Extracts only the Close column for simplicity.
Scales data to [0, 1] range using MinMaxScaler.
Returns the scaled data and the scaler (useful for inverse-transforming predictions).
Creating Sequences (Windowing)
python
Copy code
def create_sequences(dataset, window_size):
    X, y = [], []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i - window_size:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)
Windows the data: each sequence of length window_size becomes one sample (X), and the next point is the label (y).
Returns NumPy arrays for features and labels.
Splitting Train/Test
python
Copy code
def split_train_test(X, y, test_ratio):
    split_index = int(len(X) * (1 - test_ratio))
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]
    return X_train, y_train, X_test, y_test
Splits data into training and testing sets based on TEST_SPLIT_RATIO (e.g., 80% training, 20% testing).
Model Building
python
Copy code
def build_lstm_model(window_size, units=64, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=False, input_shape=(window_size, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
Creates a Sequential LSTM-based model with:
A single LSTM layer
A dropout layer for regularization
A dense layer with a single neuron (for final output)
Compiles with adam optimizer and mean_squared_error loss.
Model Training & Evaluation
In the main() function:

Download Data:
python
Copy code
df = load_stock_data(STOCK_SYMBOL, START_DATE, END_DATE)
Preprocess:
python
Copy code
scaled_data, scaler = preprocess_data(df)
Create sequences:
python
Copy code
X, y = create_sequences(scaled_data, WINDOW_SIZE)
Reshape X to (samples, window_size, 1) for LSTM input:
python
Copy code
X = X.reshape((X.shape[0], X.shape[1], 1))
Split into train/test:
python
Copy code
X_train, y_train, X_test, y_test = split_train_test(X, y, TEST_SPLIT_RATIO)
Build the model:
python
Copy code
model = build_lstm_model(WINDOW_SIZE, UNITS, DROPOUT_RATE)
Train:
python
Copy code
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)
Evaluate:
python
Copy code
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {test_loss}")
Visualization
After training, predictions are generated:

python
Copy code
predictions = model.predict(X_test)
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
A typical plot (with Matplotlib) is:

python
Copy code
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label="True")
plt.plot(predictions_rescaled, label="Predicted")
plt.title("AAPL Stock Price Prediction")
plt.xlabel("Time (Days in Test Set)")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()
Note: If you’re in a headless environment (e.g., Codespaces, Docker), no window will pop up. You can save the plot instead:

python
Copy code
plt.savefig("prediction_plot.png")
Model Saving
python
Copy code
model.save("lstm_stock_model.keras")
Saves the model in .keras format (preferred over .h5).
You can load it back later with:
python
Copy code
new_model = tf.keras.models.load_model("lstm_stock_model.keras")
Handling Visualization in Headless Environments
If you don’t see charts pop up, you’re likely on a system without a GUI (e.g., Docker container, GitHub Codespaces, SSH session). Common solutions:

Save the plot to a file:

python
Copy code
plt.savefig("prediction_plot.png")
Then open or download that file locally.

Use Jupyter notebooks with %matplotlib inline:

python
Copy code
%matplotlib inline
import matplotlib.pyplot as plt
This displays the figure inline in the notebook.

VSCode / Codespaces:

Use the Python Interactive Window (the “Jupyter” style) so plots appear inline.
Or save the plot (.png / .pdf) and open it locally or through the file explorer.
Troubleshooting
NameError: 'Sequential' is not defined
Ensure from tensorflow.keras.models import Sequential or use tf.keras.models.Sequential().
UserWarning about .h5
This is a legacy format warning; switch to .keras if you want to follow the latest recommendation.
UserWarning about input_shape
Keras suggests using a dedicated Input(shape=...) layer in Sequential models. It’s just a best practice warning; your code still works.
Cannot See Charts
Save the plot or run in a notebook environment that supports inline graphics.
Import Errors
Double-check you installed packages from requirements.txt in the same environment you’re running in.