# Combined models file
# Created by concatenating: linearlogistic.py, naivebayes.py, perceptron.py, svm.py, ensemble.py, ensemblepca.py, clustering.py
# NOTE: This file is a literal concatenation of the original scripts for convenience.

# ---------- BEGIN linearlogistic.py ----------
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



# ---------- END linearlogistic.py ----------


# ---------- BEGIN naivebayes.py ----------
## CategoricalNB 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
zoo_df= pd.read_csv('zoo.csv')
X = zoo_df.drop(['animal_name', 'class_type'], axis=1)
y = zoo_df['class_type']
# Treat 'legs' as categorical for Naive Bayes
X['legs']= X['legs'].astype(str)
# Make sure all features are category type
for col in X.columns:
X[col]= X[col].astype('category')
#Model training
X_train, X_test, y_train, y_test= train_test_split(X, y, stratify=y, test_size
model = CategoricalNB()
model.fit(X_train, y_train)
#Prediction & Evaluation
y_pred= model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n"
, confusion_matrix(y_test, y_pred))
# For multiclass ROC/AUC, get predicted probabilities
y_proba= model.predict_proba(X_test)
n_classes= len(set(y))
y_test_bin= label_binarize(y_test, classes=range(1, n_classes+1))
# Calculate macro-average ROC-AUC
auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
print("Macro-averaged ROC AUC:", auc)


                                                   ## GaussianNB
                                                   import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
classification_report,
confusion_matrix,
roc_auc_score,
roc_curve,
)
import matplotlib.pyplot as plt
# Load the dataset (update 'path/to/data.csv' as needed)
df = pd.read_csv('banknotes.csv')
# Features and label
X = df.drop('class', axis=1)
y = df['class']
# Split (stratify ensures similar class distribution)
X_train, X_test, y_train, y_test= train_test_split(
X, y, test_size=0.3, random_state=42
)
# Train GaussianNB classifier
gnb= GaussianNB()
gnb.fit(X_train, y_train)
# Predictions
y_pred= gnb.predict(X_test)
y_prob= gnb.predict_proba(X_test)[:, 1] # Probability for class 1
# Metrics
print("Classification Report:\n"
, classification_report(y_test, y_pred))
print("Confusion Matrix:\n"
, confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
# ROC Curve
fpr, tpr,
_
= roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'GNB (AUC = {roc_auc_score(y_test, y_prob):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Banknote Authentication (GNB)')
plt.legend()
plt.show()

## MultinomialNB

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
df = pd.read_csv('imdb.csv')
# Mapping sentiment to numeric labels
df['sentiment']= df['sentiment'].map({'positive': 1, 'negative': 0})
# Train-test split
X_train, X_test, y_train, y_test= train_test_split(
df['review'], df['sentiment'], test_size=0.3, stratify=df['sentiment'], random_sta
# Building a pipeline to vectorize text data and train MultinomialNB
model = Pipeline([
('vect', CountVectorizer(stop_words='english', max_df=0.95, min_df=5)), ('tfidf', TfidfTransformer()), # Convert counts to TF-IDF representation ('clf', MultinomialNB())
# Convert
(optiona
])
# Train the model
model.fit(X_train, y_train)
# Predict on test data
y_pred= model.predict(X_test)
# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n"
, classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n"
, confusion_matrix(y_test, y_pred))



## BernoulliNB

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
df = pd.read_csv('email.csv')
# Mapping categories to numeric labels
df['Category']= df['Category'].map({'ham': 0, 'spam': 1})
df = df.dropna(subset=['Category']) #dropping null values
# Train test split
X_train, X_test, y_train, y_test= train_test_split(
df['Message'], df['Category'], test_size=0.2, random_state=42, stratify=df[
# Text to binary features
vectorizer = CountVectorizer(binary=True, stop_words='english', min_df=2)
X_train_bin= vectorizer.fit_transform(X_train)
X_test_bin= vectorizer.transform(X_test)
# Model training
bnb = BernoulliNB()
bnb.fit(X_train_bin, y_train)
# Prediction & Evaluation
y_pred= bnb.predict(X_test_bin)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n"
, classification_report(y_test, y_pred, target_names
print("\nConfusion Matrix:\n"
, confusion_matrix(y_test, y_pred))




# ---------- END naivebayes.py ----------


# ---------- BEGIN perceptron.py ----------
## Manual Perceptron



import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_function
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initializing weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Update
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_function(self, x):
        return np.where(x >= 0, 1, 0)

    def decisionBoundary(self):
      x1 = np.linspace(-0.5, 1.5, 100)
      x2 = -(self.weights[0]*x1 + self.bias + 0.02) / self.weights[0]


      plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired)
      plt.plot(x1, x2, 'k--', label='Decision Boundary')
      plt.xlabel('X1')
      plt.ylabel('X2')
      plt.title('Perceptron OR Gate Decision Boundary')
      plt.legend()
      plt.show()

# Example: OR gate data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 1])

p = Perceptron()
p.fit(X, y)
predictions = p.predict(X)
print("Predictions:", predictions)

p.decisionBoundary()

## using sklearn


from sklearn.linear_model import Perceptron
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, RocCurveDisplay
import pandas as pd

df = pd.read_csv("IMDB Dataset.csv")

# Convert sentiment to binary labels: pos -> 1, neg -> 0
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X_text = df['review']
y = df['label']

#  Train/Test Split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)

# Vectorize Text (TF-IDF)
vectorizer = TfidfVectorizer(
    stop_words='english',      # remove common English stopwords
    max_features=20000          # limit features to reduce dimensionality
)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Create and Train Perceptron
clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
clf.fit(X_train, y_train)

# Predictions and Accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Some predictions vs actual
for review, actual, pred in zip(X_test_text[:5], y_test[:5], y_pred[:5]):
    print(f"\nReview: {review[:120]}...")
    print(f"Actual: {actual}, Predicted: {pred}")

# Metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

# Perceptron doesn't provide probability; use decision_function instead
y_scores = clf.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
auc_score = roc_auc_score(y_test, y_scores)
print(f"ROC AUC Score: {auc_score:.4f}")

RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score).plot()
plt.show()




### MLP 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Column names for flags dataset
columns = [
    "name", "landmass", "zone", "area", "population", "language",
    "religion", "bars", "stripes", "colours", "red", "green",
    "blue", "gold", "white", "black", "orange", "mainhue",
    "circles", "crosses", "saltires", "quarters", "sunstars",
    "crescent", "triangle", "icon", "animate", "text", "topleft",
    "botright"
]

# Load dataset
df = pd.read_csv('flag.data', names=columns)
print(df.head())
print(df.shape)
print("\n", df['religion'].value_counts())

# Features and target - predicting religion based on flag characteristics

X = df.drop(columns=['name', 'religion'])  # Remove name and target
y = df['religion']

# Encode any categorical columns
for col in X.columns:
    if X[col].dtype == 'object':
        print(f"Encoding categorical column: {col}")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# Encode target
target_enc = LabelEncoder()
y_enc = target_enc.fit_transform(y)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Different MLP architectures
architectures = {
    '1_hidden_layer': (50,),
    '2_hidden_layers': (100, 50),
    '3_hidden_layers': (150, 100, 50),
    '4_hidden_layers': (200, 150, 100, 50),
    '5_hidden_layers': (250, 200, 150, 100, 50)
}

# Train and evaluate each architecture
print("\n" + "-"*60)
print("TRAINING MLP WITH DIFFERENT ARCHITECTURES")
print("-"*60)

for name, layers in architectures.items():
    print(f"\nTraining MLP with {name}: {layers}")

    mlp = MLPClassifier(
        hidden_layer_sizes=layers,
        max_iter=2000,  # More iterations
        random_state=42,
        alpha=0.01,  # More regularization to prevent overfitting
        solver='adam',
        learning_rate_init=0.01,  # Higher learning rate
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50
    )

    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy for {name}: {acc:.4f}")
    print(f"Training converged in {mlp.n_iter_} iterations")



# ---------- END perceptron.py ----------


# ---------- BEGIN svm.py ----------
# Binary Class SVM 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve,
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
# 1. Load dataset
df = pd.read_csv('emails.csv')
# Map spam field to binary: 'spam' -> 1, 'ham' -> 0
df['label_binary']= df['spam'].map({'ham': 0, 'spam': 1})
X_raw= df['text']
y = df['spam']
# Remove null values from dataset
df_clean= df.dropna(subset=['spam', 'text'])
# 2. Text vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', lowercase
X_tfidf= vectorizer.fit_transform(X_raw)
# 3. Train-test split
X_train, X_test, y_train, y_test= train_test_split(X_tfidf, y, test_size=0.2,
# 4. Model configs with multiple kernels
models = [
{"model_num": 1, "kernel": "linear", "params": {"C": 1}},
{"model_num": 2, "kernel": "poly", "params": {"C": 1, "degree": 3, "gamma":
{"model_num": 3, "kernel": "rbf", "params": {"C": 1, "gamma": "scale"}},
{"model_num": 4, "kernel": "sigmoid", "params": {"C": 1, "gamma": "scale"}}
]
results = []
# 5. Train, evaluate, collect results
for m in models:
clf = SVC(kernel=m['kernel'],
C=m['params'].get('C', 1),
degree=m['params'].get('degree', 3),
gamma=m['params'].get('gamma', 'scale'),
probability=True)
clf.fit(X_train, y_train)
y_pred= clf.predict(X_test)
y_prob= clf.predict_proba(X_test)[:, 1]
report= classification_report(y_test, y_pred, output_dict=True)
accuracy = report['accuracy']
precision= report['1']['precision']
recall = report['1']['recall']
f1 = report['1']['f1-score']
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
results.append({
'Model': m['model_num'],
'Kernel': m['kernel'],
'Parameters': m['params'],
'Accuracy': accuracy,
'Precision': precision,
'Recall': recall,
'F1-score': f1,
'ROC AUC': roc_auc,
'Confusion Matrix': cm,
'Classifier': clf,
'Probabilities': y_prob
})
# 6. Summarize results in table
summary_df= pd.DataFrame([{k: v for k, v in r.items() if k not in ['Confusion Matrix'
print(summary_df)
# 7. Performance comparison plot
plt.figure(figsize=(10, 5))
plt.plot(summary_df['Model'], summary_df['Accuracy'], marker='o', label='Accuracy'
plt.plot(summary_df['Model'], summary_df['F1-score'], marker='s', label='F1-score'
plt.plot(summary_df['Model'], summary_df['ROC AUC'], marker='^'
, label='ROC AUC'
plt.xticks(summary_df['Model'])
plt.xlabel('Model Number')
plt.ylabel('Score')
plt.title('SVM Kernel Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()
# 8. ROC Curve and Confusion Matrix plotting
for r in results:
fpr, tpr,
_
= roc_curve(y_test, r['Probabilities'])
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'Model {r["Model"]} ({r["Kernel"]})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - Model {r["Model"]} (Kernel={r["Kernel"]})')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
plt.figure(figsize=(4, 4))
sns.heatmap(r['Confusion Matrix'], annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - Model {r["Model"]} (Kernel={r["Kernel"]})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# 9. Decision boundary plotting (optional and approximate with PCA on subset)
sample_size= 1000 # larger text dataset, smaller sample for visualization
X_vis= X_train[:sample_size].toarray()
y_vis= y_train.iloc[:sample_size]
pca = PCA(n_components=2)
X_vis_2d= pca.fit_transform(X_vis)
for m in models:
continue
clf = SVC(kernel=m['kernel'],
C=m['params'].get('C', 1),
if m['kernel'] not in ['linear', 'rbf']: # show only linear and rbf for clearer b
degree=m['params'].get('degree', 3),
gamma=m['params'].get('gamma', 'scale'))
clf.fit(X_vis_2d, y_vis)
x_min, x_max = X_vis_2d[:, 0].min()- 1, X_vis_2d[:, 0].max() + 1
y_min, y_max = X_vis_2d[:, 1].min()- 1, X_vis_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
scatter = plt.scatter(X_vis_2d[:, 0], X_vis_2d[:, 1], c=y_vis, cmap='coolwarm'
plt.title(f'Decision Boundary - SVM ({m["kernel"]} kernel)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


## Multi Class SVM


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve,
from sklearn.preprocessing import label_binarize, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
df = pd.read_csv('zoo.csv')
# Features and target
# Drop non-feature column 'animal_name'
X = df.drop(['animal_name', 'class_type'], axis=1)
# Use 'class_type' as the target variable; encode it as numeric labels
le = LabelEncoder()
y = le.fit_transform(df['class_type'])
# Binarize labels for ROC AUC
classes = np.unique(y)
y_bin= label_binarize(y, classes=classes)
# Split data
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state
# Also split y_bin to keep test labels aligned
_
, y_bin_test= train_test_split(y_bin, test_size=0.2, random_state=42)
# Normalize features
scaler = StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)
# Define SVM models with different kernels and parameters
models = [
{"model_num": 1, "kernel": "linear", "params": {"C": 1}},
{"model_num": 2, "kernel": "poly", "params": {"C": 1, "degree": 3, "gamma":
{"model_num": 3, "kernel": "rbf", "params": {"C": 1, "gamma": "scale"}},
{"model_num": 4, "kernel": "sigmoid", "params": {"C": 1, "gamma": "scale"}}
]
results = []
# Train, evaluate, and store results
for m in models:
clf = SVC(kernel=m['kernel'], C=m['params'].get('C',1), degree=m['params'].
gamma=m['params'].get('gamma','scale'), probability=True)
clf.fit(X_train_scaled, y_train)
y_pred= clf.predict(X_test_scaled)
y_prob= clf.predict_proba(X_test_scaled)
# Classification report
report= classification_report(y_test, y_pred, output_dict=True, zero_division
accuracy = report['accuracy']
precision= np.mean([report[str(c)]['precision'] for c in classes])
recall = np.mean([report[str(c)]['recall'] for c in classes])
f1 = np.mean([report[str(c)]['f1-score'] for c in classes])
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
# ROC AUC
roc_auc = roc_auc_score(y_bin_test, y_prob, multi_class='ovr')
results.append({
'Model': m['model_num'],
'Kernel': m['kernel'],
'Parameters': m['params'],
'Accuracy': accuracy,
'Precision': precision,
'Recall': recall,
'F1-score': f1,
'ROC AUC': roc_auc,
'Confusion Matrix': cm,
'Classifier': clf,
'Probabilities': y_prob
})
# Display summary table
summary_df= pd.DataFrame([{k: v for k, v in r.items() if k not in ['Confusion Matrix'
print(summary_df)
# Performance comparison plot
plt.figure(figsize=(10,5))
plt.plot(summary_df['Model'], summary_df['Accuracy'], marker='o', label='Accuracy'
plt.plot(summary_df['Model'], summary_df['F1-score'], marker='s', label='F1-score'
plt.plot(summary_df['Model'], summary_df['ROC AUC'], marker='^'
, label='ROC AUC'
plt.xticks(summary_df['Model'])
plt.xlabel('Model Number')
plt.ylabel('Score')
plt.title('SVM Kernel Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()
# ROC Curves plot
plt.figure(figsize=(12,10))
colors = ['blue', 'green', 'red', 'purple']
for i, r in enumerate(results):
for j, c in enumerate(classes):
fpr, tpr,
_
= roc_curve(y_bin_test[:, j], r['Probabilities'][:, j])
plt.plot(fpr, tpr, label=f'Model {r["Model"]} Kernel={r["Kernel"]} Class=
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves per Class for each Kernel')
plt.legend(fontsize='small', loc='lower right')
plt.grid(True)
plt.show()
# Confusion Matrix plots
for r in results:
plt.figure(figsize=(6,5))
sns.heatmap(r['Confusion Matrix'], annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - Model {r['Model']} (Kernel={r['Kernel']})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Decision boundary plot with PCA reduction to 2D
pca = PCA(n_components=2)
X_train_2d= pca.fit_transform(X_train_scaled)
X_test_2d= pca.transform(X_test_scaled)
for r in results:
clf_2d= SVC(kernel=r['Kernel'], C=r['Parameters'].get('C',1), degree=r['Parameter
gamma=r['Parameters'].get('gamma','scale')
clf_2d.fit(X_train_2d, y_train)
x_min, x_max = X_train_2d[:, 0].min()- 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min()- 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max,
Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10,8))
plt.contourf(xx, yy, Z, alpha=0.3)
scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap='Set1'
plt.title(f'Decision Boundary - Model {r['Model']} (Kernel={r['Kernel']})')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(*scatter.legend_elements(), title='Classes')
plt.show()                     


# ---------- END svm.py ----------


# ---------- BEGIN ensemble.py ----------
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


# ---------- END ensemble.py ----------


# ---------- BEGIN ensemblepca.py ----------
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('geneexpression.csv')
labels = pd.read_csv('labels.csv')

# dropping Unnamed: 0
X = df.drop(columns=['Unnamed: 0'])
y = labels['Class']

le = LabelEncoder()
y_enc = le.fit_transform(y)

#EDA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print(f"Feature data shape: {X.shape}")
print(f"Target shape: {y.shape}")

sns.countplot(x=y)
plt.title("Cancer Class Distribution")
plt.show()

missing_perc = pd.DataFrame(X).isnull().mean().mean() * 100
print(f"Percent missing values in dataset: {missing_perc:.2f}%")

gene_df = pd.DataFrame(X)
gene_means = gene_df.mean()
gene_vars = gene_df.var()
top_var_genes = gene_vars.sort_values(ascending=False).head(10)

print("Top 10 genes by variance:")
print(top_var_genes)

plt.figure(figsize=(10,6))
sns.histplot(top_var_genes, bins=10, kde=True)
plt.title("Distribution of Top 10 Most Variant Genes")
plt.xlabel("Variance")
plt.show()

# Correlation structure for top genes
corr = gene_df[top_var_genes.index].corr()
plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, cmap='vlag')
plt.title("Correlation Among Top 10 Most Variable Genes")
plt.show()

plt.figure(figsize=(7,5))
example_gene = gene_df.columns[gene_vars.argmax()]  # Most variable gene
sns.violinplot(x=y, y=gene_df[example_gene])
plt.title(f"Expression of {example_gene} Across Cancer Types")
plt.show()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='Set2')
plt.title("PCA: 2 Components")
plt.show()

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=40)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette='Set2')
plt.title("t-SNE: 2 Components")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# split
X_pca_train, X_pca_test, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)
X_tsne_train, X_tsne_test, _, _ = train_test_split(X_tsne, y, test_size=0.2, random_state=42)


def evaluate_model(model, Xtr, Xte, ytr, yte, name="Classifier"):
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)
    print(f"{name} Results:")
    print(classification_report(yte, y_pred))
    cm = confusion_matrix(yte, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.show()
    try:
        y_proba = model.predict_proba(Xte)
        auc = roc_auc_score(yte, y_proba, multi_class='ovr')
        print(f"AUC: {auc}")
    except Exception as e:
        print(f"AUC cannot be calculated for this model: {e}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model(rf, X_train, X_test, y_train, y_test, "Random Forest (All Features)")
evaluate_model(rf, X_pca_train, X_pca_test, y_train, y_test, "Random Forest (PCA Reduced)")
evaluate_model(rf, X_tsne_train, X_tsne_test, y_train, y_test, "Random Forest (t-SNE Reduced)")

# AdaBoost
ab = AdaBoostClassifier(n_estimators=100, random_state=42)
evaluate_model(ab, X_train, X_test, y_train, y_test, "AdaBoost (All Features)")
evaluate_model(ab, X_pca_train, X_pca_test, y_train, y_test, "AdaBoost (PCA Reduced)")
evaluate_model(ab, X_tsne_train, X_tsne_test, y_train, y_test, "AdaBoost (t-SNE Reduced)")

# XGBoost
xgb_clf = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42)
X_pca_train_xgb, X_pca_test_xgb, _, _ = train_test_split(X_pca, y_enc, test_size=0.2, random_state=42)
X_tsne_train_xgb, X_tsne_test_xgb, _, _ = train_test_split(X_tsne, y_enc, test_size=0.2, random_state=42)


evaluate_model(xgb_clf, X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb, "XGBoost (All Features)")
evaluate_model(xgb_clf, X_pca_train_xgb, X_pca_test_xgb, y_train_xgb, y_test_xgb, "XGBoost (PCA Reduced)")
evaluate_model(xgb_clf, X_tsne_train_xgb, X_tsne_test_xgb, y_train_xgb, y_test_xgb, "XGBoost (t-SNE Reduced)")

# classification results data from obtained values
data = {
    "S. No": list(range(1,10)),
    "Classifier": [
        "Random Forest", "Random Forest", "Random Forest",
        "AdaBoost", "AdaBoost", "AdaBoost",
        "XGBoost", "XGBoost", "XGBoost"
    ],
    "All features / Reduced Features": [
        "All features", "PCA Reduced", "t-SNE Reduced",
        "All features", "PCA Reduced", "t-SNE Reduced",
        "All features", "PCA Reduced", "t-SNE Reduced"
    ],
    "Precision": [
        1.00, 0.69, 1.00,
        0.98, 0.70, 0.47,
        1.00, 0.69, 1.00
    ],
    "Recall": [
        1.00, 0.65, 1.00,
        0.98, 0.67, 0.58,
        1.00, 0.66, 1.00
    ],
    "F1": [
        1.00, 0.67, 1.00,
        0.98, 0.68, 0.49,
        1.00, 0.68, 1.00
    ]
}

df_results = pd.DataFrame(data)
print(df_results)


# ---------- END ensemblepca.py ----------


# ---------- BEGIN clustering.py ----------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('spotifysongs.csv')

# List numerical and categorical columns
num_features = ['energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence', 'speechiness']
cat_features = ['playlist_genre', 'track_artist', 'time_signature']

# Summary statistics for numerical features
print("Numerical Features Summary:")
print(df[num_features].describe())

# Distribution plots for numerical features
for col in num_features:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Check for missing values
print("\nMissing Values by Column:")
print(df.isnull().sum())

# Correlation heatmap for numerical features
plt.figure(figsize=(8, 6))
corr = df[num_features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix - Numerical Features')
plt.show()


from sklearn.preprocessing import StandardScaler, LabelEncoder

# Data Preprocessing for Mixed Data

# Numerical features
num_features = ['energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence', 'speechiness']

# Categorical features
cat_features = ['playlist_genre', 'track_artist', 'time_signature']

# Scale numerical features
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(df[num_features])

# Encode categorical features
encoded_cats = []
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    encoded_col = le.fit_transform(df[col])
    encoded_cats.append(encoded_col)
    label_encoders[col] = le

import numpy as np
X_cat_encoded = np.array(encoded_cats).T

# Combined dataset for algorithms accepting mixed/numerical data
X_combined = np.hstack((X_num_scaled, X_cat_encoded))

# Elbow method & K-Means/K-means++ WCSS

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss_kmeans = []
wcss_kmeanspp = []
range_n_clusters = range(2, 11)

for k in range_n_clusters:
    kmeans = KMeans(n_clusters=k, init='random', random_state=42)
    kmeans.fit(X_num_scaled)
    wcss_kmeans.append(kmeans.inertia_)
    
    kmeanspp = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeanspp.fit(X_num_scaled)
    wcss_kmeanspp.append(kmeanspp.inertia_)

plt.plot(range_n_clusters, wcss_kmeans, marker='o', label='K-Means random init')
plt.plot(range_n_clusters, wcss_kmeanspp, marker='s', label='K-Means++')
plt.xlabel('Number of clusters k')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method for K-Means Variants')
plt.legend()
plt.show()

# silhouette scores for all algos

from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from kmodes.kmodes import KModes

optimal_k = 4  # selected based on elbow

# K-Means++
kmeanspp = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
labels_kmeanspp = kmeanspp.fit_predict(X_num_scaled)
score_kmeanspp = silhouette_score(X_num_scaled, labels_kmeanspp)
print("K-Means++ Silhouette:", score_kmeanspp)

# Hierarchical
hier = AgglomerativeClustering(n_clusters=optimal_k)
labels_hier = hier.fit_predict(X_num_scaled)
score_hier = silhouette_score(X_num_scaled, labels_hier)
print("Hierarchical Silhouette:", score_hier)

# DBSCAN
dbscan = DBSCAN(eps=1.3, min_samples=10)
labels_db = dbscan.fit_predict(X_num_scaled)
mask_db = labels_db != -1  # exclude noise
score_dbscan = silhouette_score(X_num_scaled[mask_db], labels_db[mask_db])
print("DBSCAN Silhouette:", score_dbscan)

# K-Modes (only categorical)
kmodes = KModes(n_clusters=optimal_k, init='Huang', n_init=5, verbose=0)
labels_kmodes = kmodes.fit_predict(X_cat_encoded)
# Silhouette not commonly used directly with K-Modes (categorical), Using other measures like cost.
print("K-Modes Cluster cost:", kmodes.cost_)


# Mode per cluster for each categorical colunmn

# Extract only categorical columns for K-Modes
df_cat = df[cat_features].copy()

# Add cluster labels to categorical dataframe
df_cat['cluster'] = labels_kmodes

def get_modes_per_cluster(df_cat, cat_features):
    modes = {}
    for col in cat_features:
        mode_per_cluster = df_cat.groupby('cluster')[col].agg(lambda x: x.mode().iloc[0])
        modes[col] = mode_per_cluster
    return pd.DataFrame(modes)

modes_table = get_modes_per_cluster(df_cat, cat_features)
print("Most Frequent Categories per Cluster (K-Modes):")
print(modes_table)

# Cluster visualization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_num_scaled)

def plot_clusters(X_pca, labels, title):
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='rainbow', alpha=0.6)
    plt.title(title)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.show()

plot_clusters(X_pca, labels_kmeanspp, 'K-Means++ Clusters')
plot_clusters(X_pca, labels_hier, 'Hierarchical Clusters')
plot_clusters(X_pca, labels_db, 'DBSCAN Clusters')


# ---------- END clustering.py ----------
