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
