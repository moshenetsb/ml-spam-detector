from preprocessing import preprocess_data
from pathlib import Path
from vectorization import vectorize_data
from training import train_models
from evaluation import evaluate_model


def main():
    data_folder_path = Path(__file__).resolve().parent.parent / 'data'

    print("=== Spam Detector ===")

    stages = [
        ("Preprocessing", preprocess_data),
        ("Wektoryzacja", vectorize_data),
        ("Trenowanie modeli (wybór najlepszego)", train_models),
        ("Ewaluacja i interpretacja", evaluate_model),
    ]

    for i, (name, function) in enumerate(stages, start=1):
        print(f"\n[{i}/{len(stages)}] {name}")
        function(data_folder_path)

    print("\n=== Zakończono ===")


if __name__ == "__main__":
    main()
