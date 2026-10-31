import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import BernoulliNB

from modeling import build_model_pipeline
from paths import APPLICATION_PATH


def compare_models(models, X_train, y_train, cv=None):
    """Compare model families using training-only CV, then fit the winner."""
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for name, model in models.items():
        scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring={"accuracy": "accuracy", "f1": "f1"}, error_score="raise",
        )
        result = {
            "name": name,
            "cv_accuracy": float(np.mean(scores["test_accuracy"])),
            "cv_f1": float(np.mean(scores["test_f1"])),
        }
        results.append(result)
        print(f"{name}: CV Accuracy = {result['cv_accuracy']:.4f}, CV F1 = {result['cv_f1']:.4f}")

    best = max(results, key=lambda result: result["cv_f1"])
    best_model = clone(models[best["name"]]).fit(X_train, y_train)
    print(f"Best model by CV: {best['name']} (CV F1: {best['cv_f1']:.4f})")
    return best_model, best["cv_f1"], best["name"], results


def train_models(base_path=APPLICATION_PATH):
    vectorized_dir = base_path / "vectorized"
    output_dir = base_path / "models"

    X_train, _, y_train, _ = joblib.load(
        vectorized_dir / "text_splits.pkl")

    best_model = build_model_pipeline(BernoulliNB(alpha=0.1))

    best_model.fit(X_train, y_train)

    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        best_model,
        output_dir / "spam_classifier_model.pkl"
    )

    return best_model


if __name__ == "__main__":
    train_models()
