# PRACTICAL 8: ML pipeline with preprocessing and automatic model selection
# Install: python -m pip install scikit-learn pandas
# Uses GridSearchCV instead of TPOT to make setup simpler in Python IDLE.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", SVC())
])

parameters = [
    {"model": [SVC()], "model__C": [0.1, 1, 10],
     "model__kernel": ["linear", "rbf"]},
    {"model": [RandomForestClassifier(random_state=42)],
     "model__n_estimators": [50, 100],
     "model__max_depth": [None, 3, 5]}
]

search = GridSearchCV(pipeline, parameters, cv=5, scoring="accuracy")
search.fit(X_train, y_train)
predictions = search.predict(X_test)

print("Best parameters:", search.best_params_)
print("Test accuracy:", round(accuracy_score(y_test, predictions), 4))
