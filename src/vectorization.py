from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def vectorize_data(base_path=Path(__file__).resolve().parent.parent / "data") -> None:
    """
    Vectorizes the preprocessed spam dataset using TF-IDF and saves the vectorizer and the train-test splits to disk
    """

    data_path = base_path / "data_processed.csv"
    output_dir = base_path / "vectorized"

    if not data_path.exists():
            raise FileNotFoundError(f"Preprocessed file not found: {data_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2, random_state=42, stratify=y,
    )

    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words="english",
        lowercase=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.pkl")
    joblib.dump(
        (X_train_tfidf, X_test_tfidf, y_train, y_test),
        output_dir / "tfidf_splits.pkl",
    )


if __name__ == "__main__":
    vectorize_data()
