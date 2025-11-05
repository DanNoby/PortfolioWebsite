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



