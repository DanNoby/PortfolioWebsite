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

