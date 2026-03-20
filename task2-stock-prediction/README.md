# Task 2: Predict Future Stock Prices (Short-Term)

---

## Objective

Use historical stock data to predict the next day's closing price using regression models.

---

## Dataset

- **Stock:** Apple Inc. (AAPL)
- **Source:** Yahoo Finance via `yfinance` Python library
- **Period:** January 2019 – December 2024
- **Features:** Open, High, Low, Close, Volume (+ engineered features)

---

## Tools & Libraries

- Python 3
- yfinance
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

---

## What the Notebook Does

1. Fetches 5 years of AAPL daily stock data via `yfinance`
2. Performs EDA — closing price trend and volume over time
3. Engineers features: intraday change, daily range, 5-day MA, 10-day MA
4. Creates target variable: next day's closing price (shifted by -1)
5. Splits data 80/20 (time-ordered, no shuffle)
6. Scales features with StandardScaler
7. Trains Linear Regression and Random Forest Regressor
8. Plots actual vs predicted prices for both models
9. Visualizes Random Forest feature importance

---

## Models Applied

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline regression assuming linear relationships |
| Random Forest Regressor | Ensemble method capturing non-linear patterns |

---

## Key Results and Findings

- Random Forest achieves lower MAE and higher R² than Linear Regression
- Moving averages (MA_5, MA_10) and current Close price are the most important features
- Both models track the general price trend well but struggle with sudden jumps
- Stock prediction from OHLCV data alone has inherent limits — news and events aren't captured

---

## How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `task2_stock_prediction.ipynb`
3. Click **Runtime → Run All**
4. `yfinance` installs automatically in the first cell

---

## File Structure

```
task2-stock-prediction/
├── task2_stock_prediction.ipynb   # Main notebook
└── README.md                      # This file
```

---

## Disclaimer

This project is for educational purposes only. Do not use these predictions for actual trading or financial decisions.
