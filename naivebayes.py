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








