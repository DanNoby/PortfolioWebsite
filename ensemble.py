# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from math import sqrt

# Load Dataset
df = pd.read_csv("Life Expectancy Data.csv")

# Data Preprocessing - drop non-numeric before correlation
df = df.drop(['Country', 'Year'], axis=1)
df = df.dropna()
le = LabelEncoder()
df['Status'] = le.fit_transform(df['Status'])

# --- Exploratory Data Analysis (EDA) ---
print("Dataset Overview:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Correlation heatmap with numeric cleaned data
plt.figure(figsize=(12,10))
correlation = df.corr()
sns.heatmap(correlation[['Life expectancy ']].sort_values(by='Life expectancy ', ascending=False), annot=True, cmap='coolwarm')
plt.title("Feature Correlations with Life Expectancy")
plt.show()

# Distribution plot of the target
plt.figure(figsize=(8,5))
sns.histplot(df['Life expectancy '], kde=True, color='green')
plt.title("Distribution of Life Expectancy")
plt.xlabel("Life Expectancy")
plt.ylabel("Frequency")
plt.show()

# Split data
X = df.drop('Life expectancy ', axis=1)
y = df['Life expectancy ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models with parameter grids for tuning
param_grids = {
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'AdaBoost': {
        'model': AdaBoostRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=42, objective='reg:squarederror'),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
    }
}

# Define RMSE function since root_mean_squared_error is not built-in
def root_mean_squared_error(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

# Parameter tuning and evaluation loop
results = []
for name, mp in param_grids.items():
    print(f"\nTuning {name}...")
    grid_search = GridSearchCV(mp['model'], mp['params'], cv=3, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append([name, mae, rmse, r2])

# Baseline Linear Regression for comparison
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results.append([
    'Linear Regression',
    mean_absolute_error(y_test, y_pred_lr),
    root_mean_squared_error(y_test, y_pred_lr),
    r2_score(y_test, y_pred_lr)
])

# Tabulate final results
results_df = pd.DataFrame(results, columns=['Model', 'MAE', 'RMSE', 'R2 Score'])
print("\nFinal Comparison:")
print(results_df)
