# DevelopersHub Corporation - AI/ML Engineering Internship

This repository contains my completed tasks for the **AI/ML Engineering Internship** at DevelopersHub Corporation.

**Intern:** Muhammad Ahtasham Ul Haq  
**GitHub:** [@mr-ahtashamulhaq](https://github.com/mr-ahtashamulhaq)  

---

## Tasks Overview

| # | Task | Dataset | Models Used | Status |
|---|------|---------|-------------|--------|
| 1 | Exploring and Visualizing the Iris Dataset | Iris Dataset (seaborn) | EDA only | ✅ Done |
| 2 | Predict Future Stock Prices (Short-Term) | Apple (AAPL) via yfinance | Linear Regression, Random Forest | ✅ Done |
| 3 | House Price Prediction | California Housing (scikit-learn) | Linear Regression, Gradient Boosting | ✅ Done |

---

## Task 1 – Exploring and Visualizing the Iris Dataset

**Folder:** `task1-iris-eda/`

Loaded and explored the classic Iris dataset using pandas, then built a full set of visualizations — pairplot scatter matrix, feature histograms with KDE curves, box plots for outlier detection, and a correlation heatmap.

**Key finding:** Petal length and petal width are the strongest separators between species. Setosa is completely distinct from the other two classes.

---

## Task 2 – Predict Future Stock Prices (Short-Term)

**Folder:** `task2-stock-prediction/`

Fetched 5 years of Apple (AAPL) daily stock data via `yfinance` and trained two models to predict the next day's closing price. Engineered additional features including 5-day and 10-day moving averages, intraday price change, and daily range.

**Key finding:** Random Forest outperformed Linear Regression. Moving averages and the current closing price carried the most predictive signal.

---

## Task 3 – House Price Prediction

**Folder:** `task6-house-price-prediction/`

Used the California Housing Dataset to predict median house values. Performed EDA including a geographic price map, trained Linear Regression and Gradient Boosting, and analyzed feature importance.

**Key finding:** Gradient Boosting significantly outperformed Linear Regression. Median income and geographic location are the two strongest drivers of house prices.

---

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- yfinance
- Jupyter Notebook (Google Colab)

---

## How to Run Any Notebook

1. Open [Google Colab](https://colab.research.google.com)
2. Upload the `.ipynb` file from the relevant task folder
3. Click **Runtime → Run All**
4. Task 2 requires an internet connection to fetch stock data — Colab handles this automatically

---

## Repository Structure

```
developershub-aiml-internship/
│
├── task1-iris-eda/
│   ├── task1_iris_eda.ipynb
│   └── README.md
│
├── task2-stock-prediction/
│   ├── task2_stock_prediction.ipynb
│   └── README.md
│
├── task6-house-price-prediction/
│   ├── task6_house_price_prediction.ipynb
│   └── README.md
│
└── README.md  ← You are here
```
