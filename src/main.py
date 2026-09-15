from preprocessing import preprocess_data
from pathlib import Path
from vectorization import vectorize_data
# from training import train_models


def main():
    data_folder_path = Path(__file__).resolve().parent.parent / 'data'

    print("=== Spam Detector ===")

    print("\n[1/3] Preprocessing")
    preprocess_data(data_folder_path)

    print("\n[2/3] Wektoryzacja")
    vectorize_data(data_folder_path)

    # print("\n[3/3] Trenowanie modeli")
    # train_models()

    print("\n=== Zakończono ===")


if __name__ == "__main__":
    main()
