import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv('glass.csv')  

# Separate features and label
X = df.drop('Type', axis=1)
y = df['Type']
df = pd.read_csv('glass.csv')

# Basic info and stats
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Class distribution plot
sns.countplot(x='Type', data=df)
plt.title('Glass Type Distribution')
plt.show()

# Feature distribution plots
for col in df.columns[:-1]:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Glass Classification with Decision Tree (Entropy and Gini)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

# Part A: Decision Tree using Entropy
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)

print("Entropy Decision Tree Accuracy:", accuracy_score(y_test, y_pred_entropy))
print("Classification Report (Entropy):\n", classification_report(y_test, y_pred_entropy))

# Visualization
plt.figure(figsize=(18,8))
plot_tree(dt_entropy, feature_names=X.columns, class_names=[str(c) for c in sorted(y.unique())], filled=True, max_depth=4)
plt.title('Decision Tree (Entropy)', fontsize=18)
plt.show()

# Part B: Decision Tree using Gini
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)

print("Gini Decision Tree Accuracy:", accuracy_score(y_test, y_pred_gini))
print("Classification Report (Gini):\n", classification_report(y_test, y_pred_gini))

# Visualization
plt.figure(figsize=(18,8))
plot_tree(dt_gini, feature_names=X.columns, class_names=[str(c) for c in sorted(y.unique())], filled=True, max_depth=4)
plt.title('Decision Tree (Gini)', fontsize=18)
plt.show()

# Parameter Optimization (Grid Search)
param_grid = {
    'max_depth': [3, 4, 5, 6, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid = GridSearchCV(DecisionTreeClassifier(criterion='entropy', random_state=42),
                    param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print('Best params (entropy):', grid.best_params_)
print('Best cross-validation accuracy:', grid.best_score_)
best_tree = grid.best_estimator_

# Feature Importance
feature_importances = pd.Series(best_tree.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importances:\n", feature_importances)

# Visualize optimal tree
plt.figure(figsize=(18,8))
plot_tree(best_tree, feature_names=X.columns, class_names=[str(c) for c in sorted(y.unique())], filled=True, max_depth=4)
plt.title('Optimized Decision Tree (Entropy)', fontsize=18)
plt.show()


# Regression tree

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score,  root_mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("SeoulBikeData.csv")
  # Replace with actual path

df.columns = df.columns.str.strip()

print(df.info())
print(df.isnull().sum())

# Statistical summary
print(df.describe())

# Target variable distribution
sns.histplot(df['Rented Bike Count'], bins=30, kde=True)
plt.title('Distribution of Rented Bike Count')
plt.xlabel('Bike Count')
plt.ylabel('Frequency')
plt.show()

# Categorical feature counts
for col in ['Seasons', 'Holiday', 'Functioning Day']:
    sns.countplot(x=col, data=df)
    plt.title(f'Count of {col}')
    plt.show()

# Drop non-numeric columns before correlation
numeric_df = df.drop(columns=['Date', 'Seasons', 'Holiday', 'Functioning Day'])

# Calculate correlation matrix
corr = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f',
            square=True, cbar=True, linewidths=0.5)
plt.title('Correlation Heatmap (Numeric Features)')
plt.show()


# Pairplot for selected features against target
sns.pairplot(df, x_vars=['Temperature(C)', 'Humidity(%)', 'Wind speed (m/s)'],
             y_vars='Rented Bike Count', height=4, kind='scatter')
plt.suptitle('Scatter plots of Key Features vs Rented Bike Count')
plt.show()
# Convert categorical columns using one-hot encoding
categorical_cols = ['Seasons', 'Holiday', 'Functioning Day']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Features and target
X = df.drop(['Date' , 'Rented Bike Count'], axis=1)  # Remove date and target
y = df['Rented Bike Count']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Regression Tree (CART)
reg_tree = DecisionTreeRegressor(random_state=42)
reg_tree.fit(X_train, y_train)

# Predict and evaluate
y_pred = reg_tree.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print('RMSE:', rmse)
print("R2 Score:", r2_score(y_test, y_pred))

# Visualize tree (limit depth for clarity)
plt.figure(figsize=(20,10))
plot_tree(reg_tree, feature_names=X.columns, filled=True, max_depth=3)
plt.title('Regression Tree for Bike Sharing Demand')
plt.show()

# Hyperparameter tuning (optional)
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
best_tree = grid.best_estimator_

print("Best Params:", grid.best_params_)

# Evaluate tuned model
y_pred_best = best_tree.predict(X_test)
print("Tuned RMSE:", root_mean_squared_error(y_test, y_pred_best))
print('Tuned RMSE:', rmse)
print("Tuned R2 Score:", r2_score(y_test, y_pred_best))

# Visualize tuned tree
plt.figure(figsize=(20,10))
plot_tree(best_tree, feature_names=X.columns, filled=True, max_depth=3)
plt.title('Optimized Regression Tree for Bike Sharing Demand')
plt.show()
