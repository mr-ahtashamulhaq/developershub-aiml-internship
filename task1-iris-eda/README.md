# Task 1: Exploring and Visualizing the Iris Dataset

---

## Objective

Load, inspect, and visualize the classic Iris dataset to understand data trends, feature distributions, and relationships between variables.

---

## Dataset

- **Name:** Iris Dataset  
- **Source:** Loaded directly via `seaborn.load_dataset('iris')` — no file download needed  
- **Size:** 150 rows × 5 columns  
- **Features:** sepal_length, sepal_width, petal_length, petal_width, species  
- **Classes:** Setosa, Versicolor, Virginica (50 samples each)

---

## Tools & Libraries

- Python 3
- pandas
- numpy
- matplotlib
- seaborn

---

## What the Notebook Does

1. Loads the dataset using pandas/seaborn
2. Inspects shape, column names, data types, and first rows
3. Generates summary statistics with `.info()` and `.describe()`
4. Checks for missing values and class distribution
5. Creates a pairplot scatter matrix showing all feature relationships
6. Plots histograms with KDE curves for all 4 features
7. Generates box plots to identify outliers per species
8. Builds a correlation heatmap

---

## Models Applied

No model training in this task — this is purely exploratory data analysis (EDA).

---

## Key Results and Findings

- Dataset is clean — zero missing values
- Classes are perfectly balanced (50 samples each)
- Petal length and petal width are the strongest separators between species
- Sepal features alone cannot cleanly distinguish versicolor from virginica
- Petal length and petal width are highly correlated (r ≈ 0.96)
- Minor outliers exist in sepal width for setosa

---

## How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `task1_iris_eda.ipynb`
3. Click **Runtime → Run All**
4. All plots will render inline and PNG files will be saved

---

## File Structure

```
task1-iris-eda/
├── task1_iris_eda.ipynb   # Main notebook
└── README.md              # This file
```
