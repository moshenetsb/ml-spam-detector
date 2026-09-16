from pathlib import Path

from sklearn.naive_bayes import BernoulliNB
import joblib


def train_models(base_path=Path(__file__).resolve().parent.parent / "data"):
    vectorized_dir = base_path / "vectorized"
    output_dir = base_path / "models"

    X_train_tfidf, _, y_train, _ = joblib.load(
        vectorized_dir / "tfidf_splits.pkl")

    best_model = BernoulliNB(alpha=0.1)

    best_model.fit(X_train_tfidf, y_train)

    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        best_model,
        output_dir / "spam_classifier_model.pkl"
    )

    return best_model


if __name__ == "__main__":
    train_models()
