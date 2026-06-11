# 📈 Stock Price Prediction using LSTM Neural Networks — The Coca-Cola Company (KO)

## 📖 Project Overview

This project explores the application of Long Short-Term Memory (LSTM) neural networks for financial time series forecasting. The objective is to predict the next-day closing price of **The Coca-Cola Company (KO)** stock using historical market data and technical indicators derived through feature engineering.

The project follows a complete machine learning workflow, including:

* Financial data acquisition using Yahoo Finance.
* Feature engineering with technical analysis indicators.
* Data preprocessing and normalization.
* Sequence generation for time series forecasting.
* LSTM neural network design and training.
* Model evaluation using multiple regression metrics.
* Next-day stock price forecasting.

The goal is to evaluate how effectively an LSTM model can learn temporal patterns from historical stock market data and generate accurate future price predictions.

---

# 📊 Dataset

Historical stock market data was collected using the **yfinance** Python library.

### Stock

* **Ticker:** KO
* **Company:** The Coca-Cola Company

### Time Period

* Start Date: **2020-01-01**
* End Date: **2026-01-01**

### Retrieved Variables

The following daily trading information was downloaded:

* Open
* High
* Low
* Close
* Volume

These variables form the foundation for the feature engineering process.

---

# ⚙️ Feature Engineering

To improve the predictive capabilities of the model, several technical indicators were calculated from the original market data.

## Moving Averages (MA)

Moving averages smooth short-term price fluctuations and help identify trends.

### MA_5

5-day moving average:

[
MA_5 = \frac{1}{5}\sum_{i=1}^{5} Close_i
]

### MA_20

20-day moving average.

### MA_30

30-day moving average.

---

## Bollinger Bands

Bollinger Bands measure market volatility and identify potential overbought or oversold conditions.

### Upper Band

[
BB_{upper} = MA_{20} + 2\sigma
]

### Lower Band

[
BB_{lower} = MA_{20} - 2\sigma
]

where:

* (MA_{20}) = 20-day moving average
* (\sigma) = rolling standard deviation

---

## RSI (Relative Strength Index)

RSI measures the speed and magnitude of recent price changes.

Range:

* RSI > 70 → Overbought
* RSI < 30 → Oversold

Formula:

[
RSI = 100 - \frac{100}{1 + RS}
]

where:

[
RS = \frac{\text{Average Gain}}{\text{Average Loss}}
]

---

## MACD (Moving Average Convergence Divergence)

MACD is a trend-following momentum indicator.

Computed as:

[
MACD = EMA_{12} - EMA_{26}
]

Signal Line:

[
Signal = EMA_9(MACD)
]

Histogram:

[
Histogram = MACD - Signal
]

---

## Volatility

Volatility was calculated using the rolling standard deviation of daily returns over a 21-day window.

This indicator measures the magnitude of price fluctuations.

---

## Daily Return

Daily Return represents the percentage change between consecutive closing prices.

[
Return_t = \frac{Close_t - Close_{t-1}}{Close_{t-1}}
]

---

# 🎯 Final Feature Selection

After several experiments and feature selection tests, the following variables were retained as model inputs:

```python
feature_cols = [
    'Open',
    'High',
    'Low',
    'Volume',
    'MA_5',
    'MA_20',
    'MA_30',
    'Daily_return',
    'Volatility',
    'RSI'
]

target_col = 'Close'
```

The target variable is the stock closing price (**Close**).

---

# 📂 Temporal Data Split

To avoid data leakage and preserve the chronological structure of the financial time series, a temporal split was used.

| Dataset    | Percentage |
| ---------- | ---------- |
| Training   | 90%        |
| Validation | 5%         |
| Test       | 5%         |

This ensures that future information is never used during training.

---

# 🔄 Data Normalization

Feature scaling was performed using **StandardScaler** from Scikit-Learn.

Standardization transforms each feature into a distribution with:

* Mean = 0
* Standard Deviation = 1

This improves neural network convergence and training stability.

```python
from sklearn.preprocessing import StandardScaler
```

---

# ⏳ Sequence Generation

LSTM networks require sequential input data.

A rolling window approach was used with:

```python
SEQUENCE_LENGTH = 30
```

Meaning that each prediction is generated using the previous **30 trading days**.

Example:

```text
Days 1-30  -> Predict Day 31
Days 2-31  -> Predict Day 32
Days 3-32  -> Predict Day 33
...
```

Input shape:

```python
(samples, 30, features)
```

---

# 🧠 LSTM Neural Network Architecture

The final model was implemented using TensorFlow.

Architecture:

```text
LSTM(64, return_sequences=True)
Dropout(0.2)

LSTM(32)
Dropout(0.2)

Dense(16, activation='relu')

Dense(1)
```

### Optimizer

```python
Adam(learning_rate=0.001)
```

### Loss Function

```python
Huber Loss
```

Huber Loss was chosen because it is more robust to outliers than Mean Squared Error (MSE).

---

# 📈 Model Performance

The model was evaluated on the test dataset containing completely unseen observations.

## Test Metrics

```text
MAE        : $0.51
RMSE       : $0.70
Maximum Error : $2.84
R²         : 0.8892
MAPE       : 0.74%
```

---

# 🔎 Results Interpretation

### Mean Absolute Error (MAE)

The model misses the actual closing price by approximately:

```text
$0.51
```

on average.

---

### Root Mean Squared Error (RMSE)

The average prediction error remains below:

```text
$1.00
```

even when larger errors receive greater penalization.

---

### Maximum Error

The largest observed prediction error was:

```text
$2.84
```

which indicates good stability and no extreme forecasting failures.

---

### R² Score

```text
R² = 0.8892
```

The model explains approximately:

```text
88.92%
```

of the variance observed in the test dataset.

---

### Mean Absolute Percentage Error (MAPE)

```text
MAPE = 0.74%
```

The average prediction error is below 1%, demonstrating strong forecasting accuracy.

---

# 🎯 Next-Day Forecast

Using the latest available market information, the model produced the following forecast:

```text
Date of latest available data : 2025-12-31
Predicted next-day close      : $69.29
```

---

# 🛠️ Tech Stack

### Programming Language

* Python

### Data Acquisition

* yfinance

### Data Manipulation

* pandas
* numpy

### Data Visualization

* matplotlib
* seaborn

### Machine Learning

* scikit-learn

### Deep Learning

* TensorFlow

### Development Environment

* Jupyter Notebook
* VS Code

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/lorenzoperry77/Projetos.git

cd "Stock Price Prediction"
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Launch Jupyter Notebook or VSCode:

```bash
jupyter notebook
```

Open:

```text
ko_stock_prediction.ipynb
```

Run all notebook cells sequentially.

The notebook will:

1. Download stock market data.
2. Generate technical indicators.
3. Preprocess and normalize the data.
4. Create time-series sequences.
5. Train the LSTM model.
6. Evaluate model performance.
7. Generate visualizations.
8. Produce a next-day stock price forecast.

---

# 📚 Conclusion

This project demonstrates how LSTM neural networks can be applied to financial time series forecasting using historical market data and technical indicators.

The final model achieved:

* MAE = $0.51
* RMSE = $0.70
* MAPE = 0.74%
* R² = 0.8892

These results indicate that the model successfully captures a large portion of the underlying price dynamics of The Coca-Cola Company (KO), producing accurate short-term forecasts while maintaining strong generalization performance on unseen data.
