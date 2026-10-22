# PRACTICAL 10A: Train and save a machine-learning model for deployment
# Install: python -m pip install scikit-learn joblib
# Run this file first to create iris_model.joblib.

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Test accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 4))
joblib.dump(model, "iris_model.joblib")
print("Saved model as iris_model.joblib")
print("Class names:", list(data.target_names))
