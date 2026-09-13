import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def main():
    base_path = Path(__file__).resolve().parent.parent
    data_path = base_path / 'etap1' / 'data' / 'data_processed.csv'
    output_dir = base_path / 'etap2' / 'data'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Wczytywanie danych")
    df = pd.read_csv(data_path)

    df = df.dropna(subset=['text', 'label'])

    print("Generowanie wykresu")
    plt.figure(figsize=(6, 4))
    df['label'].value_counts().plot(kind='bar', color=['#2ca02c', '#d62728'])
    plt.title('Rozkład klas w zbiorze (Ham vs Spam)')
    plt.xlabel('Klasa')
    plt.ylabel('Liczba wiadomości')
    plt.xticks(rotation=0)
    plt.tight_layout()

    plot_path = output_dir / 'class_distribution.png'
    plt.savefig(plot_path)
    print(f"Wykres zapisano w: {plot_path}")

    X = df['text']
    y = df['label']

    print("Dzielenie danych na zbiór treningowy i testowy")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Eksportowanie plików konfiguracyjnych")
    joblib.dump(vectorizer, output_dir / 'tfidf_vectorizer.pkl')
    joblib.dump((X_train_tfidf, X_test_tfidf, y_train, y_test), output_dir / 'vectorized_data.pkl')

    print("\nEtap 2 Sukces")


if __name__ == "__main__":
    main()