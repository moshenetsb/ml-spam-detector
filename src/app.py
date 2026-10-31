import argparse
from pathlib import Path

import joblib

from paths import APPLICATION_PATH


def load_app_assets(base_path=APPLICATION_PATH, model_path=None):
    model_path = Path(model_path) if model_path is not None else base_path / "models" / "spam_classifier_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run src/main.py or specify --model.")

    return joblib.load(model_path)


def main():
    parser = argparse.ArgumentParser(description="Spam detector (console)")
    parser.add_argument("--model", type=Path, help="Path to a trained raw-text pipeline")
    args = parser.parse_args()
    print("Ładowanie modelu...")
    model = load_app_assets(model_path=args.model)
    print("Model gotowy do pracy! Wpisz wiadomość, aby sprawdzić, czy to spam (wpisz 'exit', aby wyjść).\n")

    while True:
        text = input("Treść wiadomości: ")
        if text.lower() == 'exit':
            break

        if not text.strip():
            print("Wiadomość nie może być pusta.\n")
            continue

        prediction = model.predict([text])[0]

        result = "Zwykła wiadomość " if prediction == 0 else "SPAM!!!"
        print(f"Wynik: {result}\n")


if __name__ == "__main__":
    main()
