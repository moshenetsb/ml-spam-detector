import csv
from pathlib import Path

import pandas as pd

from paths import APPLICATION_PATH, RAW_DATA_PATH


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """Reads the raw spam dataset from a file and returns it as a pandas DataFrame."""
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    return pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["label", "text"],
        quoting=csv.QUOTE_NONE
    )


def save_processed_data(df: pd.DataFrame, output_path: Path) -> None:
    """Saves the processed DataFrame to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Removes missing/blank texts, invalid labels and duplicates, then maps labels to binary values."""
    df = df.dropna(subset=["text", "label"])
    df = df.loc[
        df["label"].isin(["ham", "spam"])
        & df["text"].str.strip().str.len().gt(0)
    ].drop_duplicates().copy()
    df["label"] = df["label"].map({'spam': 1, "ham": 0})
    return df


def preprocess_data(base_path=APPLICATION_PATH, data_path=RAW_DATA_PATH) -> pd.DataFrame:
    """
    Preprocesses the spam dataset by reading the raw data, removing missing values and duplicates,
    and saving the cleaned data to a CSV file
    """

    output_path = base_path / "data_processed.csv"

    raw_spam_df = load_raw_data(data_path)
    spam_df = clean_dataset(raw_spam_df)
    save_processed_data(spam_df, output_path)

    return spam_df


if __name__ == "__main__":
    preprocess_data()
