from pathlib import Path
import joblib


def load_app_assets(base_path=Path(__file__).resolve().parent.parent / "data") -> tuple:
    vectorizers_dir = base_path / "vectorized"
    models_dir = base_path / "models"

    vectorizer = joblib.load(vectorizers_dir / "tfidf_vectorizer.pkl")
    model = joblib.load(models_dir / "spam_classifier_model.pkl")

    return vectorizer, model


def main():
    print("Ładowanie modelu...")
    vectorizer, model = load_app_assets()
    print("Model gotowy do pracy! Wpisz wiadomość, aby sprawdzić, czy to spam (wpisz 'exit', aby wyjść).\n")

    while True:
        text = input("Treść wiadomości: ")
        if text.lower() == 'exit':
            break

        if not text.strip():
            print("Wiadomość nie może być pusta.\n")
            continue

        text_tfidf = vectorizer.transform([text])

        prediction = model.predict(text_tfidf)[0]

        result = "Zwykła wiadomość " if prediction == 0 else "SPAM!!!"
        print(f"Wynik: {result}\n")


if __name__ == "__main__":
    main()
