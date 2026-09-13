import joblib
from pathlib import Path


def test_pkl_files():
    base_path = Path(__file__).resolve().parent
    vectorizer_path = base_path / 'data' / 'tfidf_vectorizer.pkl'
    data_path = base_path / 'data' /'vectorized_data.pkl'

    print("Wczytywanie plików\n")

    vectorizer = joblib.load(vectorizer_path)
    print("Weryfikacja TF-IDF")
    print(f"Typ obiektu: {type(vectorizer)}")
    print(f"Rozmiar słownika: {len(vectorizer.vocabulary_)}\n")

    X_train, X_test, y_train, y_test = joblib.load(data_path)
    print("Weryfikacja Danych")
    print(f"x_train (wiadomości treningowe): {X_train.shape}")
    print(f"y_train (etykiety treningowe): {y_train.shape}")
    print(f"x_test (wiadomości testowe): {X_test.shape}")
    print(f"y_test (etykiety testowe): {y_test.shape}\n")

    if X_train.shape[0] == y_train.shape[0] and X_test.shape[0] == y_test.shape[0]:
        print("Liczba wierszy w macierzach x i y zgadza się")
    else:
        print("Liczba wierszy w macierzach X i y nie zgadza się")


if __name__ == "__main__":
    test_pkl_files()