# Universal preamble
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = pd.read_excel("data.xlsx")
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
  df[col] = le.fit_transform(df[col])
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#OUTPUT / PLOTTING PREAMBLE
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
plt.figure(figsize=(6,4))
sns.histplot(df['target'], kde=True, color='green')
plt.title("Target Distribution")
plt.show()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# LINEAR / MULTIPLE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

#NAIVE BAYES (Gaussian / Multinomial / Bernoulli / Categorical)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

#PERCEPTRON / MLP
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(50,20), max_iter=500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#SVM
from sklearn.svm import SVC
model = SVC(kernel='rbf')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#CLUSTERING
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from kmodes.kmodes import KModes
model = KMeans(n_clusters=3, init='k-means++')
labels = model.fit_predict(X)
model = KModes(n_clusters=3, init='Huang')
labels = model.fit_predict(X)
model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X) 
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Assume X is your feature matrix | Dendogram
linked = linkage(X, method='ward')   # or 'complete', 'average', 'single'
plt.figure(figsize=(8,5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

#ENSEMBLES
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
model = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

#PCA + ENSEMBLE
from sklearn.decomposition import PCA
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
model = RandomForestClassifier()
model.fit(X_pca, y)

#METRICS
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))