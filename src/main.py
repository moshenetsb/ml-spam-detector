from evaluation import evaluate_model
from paths import APPLICATION_PATH
from preprocessing import preprocess_data
from training import train_models
from vectorization import vectorize_data


def main():
    data_folder_path = APPLICATION_PATH

    print("=== Spam Detector ===")

    stages = [
        ("Preprocessing", preprocess_data),
        ("Wektoryzacja", vectorize_data),
        ("Trenowanie modelu bazowego", train_models),
        ("Ewaluacja i interpretacja", evaluate_model),
    ]

    for i, (name, function) in enumerate(stages, start=1):
        print(f"\n[{i}/{len(stages)}] {name}")
        function(data_folder_path)

    print("\n=== Zakończono ===")


if __name__ == "__main__":
    main()
