## Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
fish = pd.read_csv('Fish.csv')
fish.head()

fish.describe()
fish.info()
# Visualizing pairplot for target variable correlations
sns.pairplot(fish, hue="Species")
plt.show()
# Checking for missing values
fish.isnull().sum()

# Selecting a single feature (e.g., Length1)
X = fish[['Length1']]
y = fish['Weight']
# Train-test split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state
# Model fitting
slr = LinearRegression()
slr.fit(X_train, y_train)
# Prediction
y_pred_slr= slr.predict(X_test)
# Plotting: Regression Line
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_slr, color='red', linewidth=2, label='Prediction')
plt.xlabel('Length1')
plt.ylabel('Weight')
plt.title('Simple Linear Regression: Weight vs Length1')

                                                   residuals_slr= y_test- y_pred_slr
plt.figure(figsize=(8,5))
sns.residplot(x=y_pred_slr, y=residuals_slr, lowess=True, color="purple")
plt.xlabel("Predicted Weight")
plt.ylabel("Residuals")
plt.title("Residuals Plot: SLR")
plt.show()

       ## MLR Stuff                                # Coefficients for each feature
importance= pd.Series(mlr.coef_, index=X.columns)
plt.figure(figsize=(8,5))
importance.plot(kind='bar', color='teal')
plt.title("Feature Importances (MLR Coefficients)")
plt.ylabel("Coefficient Value")

                                                   plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_mlr, color='green')
plt.xlabel("Actual Weight")
plt.ylabel("Predicted Weight")
plt.title("Actual vs Predicted: Multiple Linear Regression")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red'
plt.show()

residuals_mlr= y_test- y_pred_mlr
plt.figure(figsize=(8,5))
sns.residplot(x=y_pred_mlr, y=residuals_mlr, lowess=True, color="orange")
plt.xlabel("Predicted Weight")
plt.ylabel("Residuals")
plt.title("Residuals Plot: MLR")
plt.show()




### Logistic Regression

# Binary class 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('emails.csv')
X_text= df['text'] # Holding the email text
y = df['spam']
# Split dataset
X_train_text, X_test_text, y_train, y_test= train_test_split(X_text, y, test_size
# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=5)
X_train= vectorizer.fit_transform(X_train_text)
X_test= vectorizer.transform(X_test_text)
# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Predict
y_pred= model.predict(X_test)
y_pred_proba= model.predict_proba(X_test)[:, 1]
# Evaluation
print(classification_report(y_test, y_pred))
# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
# ROC Curve
fpr, tpr,
_
= roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC={roc_auc_score(y_test, y_pred_proba):
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# Multi Class Logistic Regression

import pandas as pd
# Defining column names
cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
# Loading the dataset
df = pd.read_csv('car.data', names=cols)
print(df.head())
print('\nClass distribution:\n'
, df['class'].value_counts())
# Encoding categorical features
from sklearn.preprocessing import LabelEncoder
# Encode all columns except for features that are already numeric
label_encoders= {}
for col in df.columns:
le = LabelEncoder()
df[col]= le.fit_transform(df[col])
label_encoders[col]= le
print(df.head())
# Training
from sklearn.model_selection import train_test_split
X = df.drop(columns=['class'])
y = df['class']
X_train, X_test, y_train, y_test= train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y)
print('Training set size:', X_train.shape)
print('Test set size:', X_test.shape)
from sklearn.linear_model import LogisticRegression
# Use multinomial option for multi-class; increase max_iter if it doesn't converge
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
y_pred= model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Classification report
print("Classification Report:\n"
, classification_report(y_test, y_pred, zero_division
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
xticklabels=label_encoders['class'].classes_,
yticklabels=label_encoders['class'].classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
# Binarize labels for ROC
n_classes= len(df['class'].unique())
y_test_bin= label_binarize(y_test, classes=range(n_classes))
y_score = model.predict_proba(X_test)
plt.figure(figsize=(8, 6))
for i in range(n_classes):
fpr, tpr,
_
= roc_curve(y_test_bin[:, i], y_score[:, i])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'Class {label_encoders["class"].classes_[i]} (AUC=
plt.plot([0, 1], [0, 1], 'k--') # baseline
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (One-vs-Rest)')
plt.legend()
plt.show()


