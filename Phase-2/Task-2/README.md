# Task 2: End-to-End ML Pipeline – Customer Churn Prediction

**DevelopersHub Corporation – AI/ML Engineering Internship**

## Objective

Build a reusable, production-ready ML pipeline to predict customer churn using the Telco Churn dataset, following best practices for preprocessing, model training, hyperparameter tuning, and export.

## Dataset

Telco Customer Churn Dataset – 7,043 customers, 20 features, binary target (Churn: Yes/No).

- Source: [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Local copy: `telco_churn.csv` (included in this repo)

## Methodology / Approach

### 1. Preprocessing Pipeline
- Dropped `customerID` (non-feature identifier)
- Converted `TotalCharges` from string to numeric; filled 11 missing values with median
- Encoded target: `No → 0`, `Yes → 1`
- `ColumnTransformer` applies:
  - `StandardScaler` on numeric features (tenure, MonthlyCharges, TotalCharges, SeniorCitizen)
  - `OneHotEncoder` (drop='first') on 15 categorical features

### 2. Models Trained
- **Logistic Regression** – with `class_weight='balanced'` to handle class imbalance
- **Random Forest** – with `class_weight='balanced'`

### 3. Hyperparameter Tuning
- `GridSearchCV` with 5-fold `StratifiedKFold`, scoring on F1
- LR grid: `C` in [0.01, 0.1, 1, 10], solver in ['lbfgs', 'liblinear']
- RF grid: `n_estimators`, `max_depth`, `min_samples_split`

### 4. Evaluation Metrics
- Accuracy, F1-Score, ROC-AUC
- Confusion Matrix, Classification Report, ROC Curve

### 5. Export
- Best pipeline exported with `joblib` → `churn_pipeline.joblib`
- Load and use: `pipeline = joblib.load('churn_pipeline.joblib'); pipeline.predict(raw_df)`

## Key Results

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression (Tuned) | 0.61 | 0.38 | 0.69 |
| Random Forest (Tuned) | 0.78 | 0.27 | 0.68 |

**Best model selected:** Logistic Regression (higher ROC-AUC and F1)

## Key Observations

- Class imbalance (~17% churn rate) was the main challenge. `class_weight='balanced'` improved recall on the minority class significantly.
- Top churn drivers: `MonthlyCharges`, `TotalCharges`, contract type (month-to-month), and `tenure`.
- The full pipeline (preprocessing + model) is encapsulated in a single `.joblib` file — no separate preprocessing needed at inference time.
- Logistic Regression outperforms Random Forest on F1 here because it handles the imbalance better with balanced weighting.

## Files

```
├── task2_churn_pipeline.ipynb   # Main notebook
├── telco_churn.csv              # Dataset
├── churn_pipeline.joblib        # Exported best pipeline
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
jupyter notebook task2_churn_pipeline.ipynb
```