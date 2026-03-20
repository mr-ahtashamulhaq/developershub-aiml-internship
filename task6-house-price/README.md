# Task 6: House Price Prediction

---

## Objective

Predict house prices using property features such as size, number of bedrooms, average rooms, income, and location.

---

## Dataset

- **Name:** California Housing Dataset
- **Source:** Built into scikit-learn (`fetch_california_housing`) — no file download needed
- **Size:** 20,640 rows × 9 columns
- **Features:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- **Target:** MedHouseVal — Median house value in $100,000 units

---

## Tools & Libraries

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

---

## What the Notebook Does

1. Loads the California Housing Dataset via scikit-learn
2. Inspects shape, data types, and checks for missing values
3. Plots target distribution (house price histogram)
4. Generates a correlation heatmap across all features
5. Scatter plot: Median Income vs House Value
6. Geographic price map: latitude/longitude colored by price
7. Splits data 80/20 and scales features with StandardScaler
8. Trains Linear Regression and Gradient Boosting Regressor
9. Plots actual vs predicted prices (scatter) for both models
10. Visualizes feature importance from Gradient Boosting
11. Compares both models with MAE, RMSE, and R²

---

## Models Applied

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline — assumes linear relationships between features and price |
| Gradient Boosting Regressor | Ensemble of decision trees — handles non-linear patterns |

---

## Key Results and Findings

- Gradient Boosting significantly outperforms Linear Regression on this dataset
- Median income is the single most important predictor of house price
- Location (Latitude + Longitude) together carry strong predictive power
- Dataset has a price cap at $500,000 which creates a ceiling visible in scatter plots
- No missing values — dataset is clean and requires minimal preprocessing

---

## How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `task6_house_price_prediction.ipynb`
3. Click **Runtime → Run All**
4. No external downloads needed — dataset loads from scikit-learn directly

---

## File Structure

```
task6-house-price-prediction/
├── task6_house_price_prediction.ipynb   # Main notebook
└── README.md                            # This file
```
