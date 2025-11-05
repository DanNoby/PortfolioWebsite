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
plt.title(f'Confusion Matrix - Model {r["Model"]} (Kernel={r["Kernel"]})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Decision boundary plot with PCA reduction to 2D
pca = PCA(n_components=2)
X_train_2d= pca.fit_transform(X_train_scaled)
X_test_2d= pca.transform(X_test_scaled)
for r in results:
clf_2d= SVC(kernel=r['Kernel'], C=r['Parameters'].get('C',1), degree=r['Parameter
gamma=r['Parameters'].get('gamma','scale'))
clf_2d.fit(X_train_2d, y_train)
x_min, x_max = X_train_2d[:, 0].min()- 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min()- 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max,
Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10,8))
plt.contourf(xx, yy, Z, alpha=0.3)
scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap='Set1'
plt.title(f'Decision Boundary - Model {r["Model"]} (Kernel={r["Kernel"]})')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(*scatter.legend_elements(), title='Classes')
plt.show()                     
