from pathlib import Path
import pandas as pd


def preprocess_data(base_path=Path(__file__).resolve().parent.parent / 'data') -> pd.DataFrame:
    """
    Preprocesses the spam dataset by reading the raw data, removing missing values and duplicates,
    and saving the cleaned data to a CSV file
    """

    data_path = base_path / "data_raw"
    output_path = base_path / "data_processed.csv"

    spam_df = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["label", "text"],
    )

    spam_df = spam_df.dropna(subset=["text", "label"])
    spam_df = spam_df.drop_duplicates()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    spam_df.to_csv(output_path, index=False)

    return spam_df


if __name__ == "__main__":
    preprocess_data()
