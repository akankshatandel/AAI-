# PRACTICAL 9B: Evolutionary model search using TPOT
# Install: python -m pip install tpot scikit-learn
# TPOT may take several minutes. If installation fails, use Practical 8.

from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

model = TPOTClassifier(
    generations=3,
    population_size=10,
    random_state=42,
    verbosity=2,
    n_jobs=1
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Test accuracy:", round(accuracy_score(y_test, predictions), 4))
model.export("best_model_pipeline.py")
print("Best pipeline saved to best_model_pipeline.py")
