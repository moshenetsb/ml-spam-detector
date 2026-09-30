import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(df, feature):

    plt.figure(figsize=(10, 5))

    sns.histplot(df[feature].dropna(), kde=True)

    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(f"Distribution of {feature}")
    plt.show()


def plot_message_length(df):
    """Show the distribution of message lengths for ham and spam."""
    
    plt.figure(figsize=(10, 5))
    
    plt.hist(
        df.loc[df["label"] == "ham", "message_length"],
        bins=40,
        alpha=0.6,
        label="ham",
    )

    plt.hist(
        df.loc[df["label"] == "spam", "message_length"],
        bins=40,
        alpha=0.6,
        label="spam",
    )
    
    plt.xlabel("Message length (characters)")
    plt.ylabel("Number of messages")
    plt.title("Distribution of message lengths")
    plt.legend()
    plt.show()


def plot_uppercase_ratio(df):
    """Show the proportion of uppercase letters for ham and spam."""

    plt.figure(figsize=(8, 5))

    df.boxplot(column="uppercase_ratio", by="label")

    plt.suptitle("")
    plt.title("Proportion of uppercase letters in messages")
    plt.xlabel("Message type")
    plt.ylabel("Uppercase letter ratio")
    plt.show()


def plot_url_count(df):
    """Show the number of URLs for ham and spam."""

    plt.figure(figsize=(8, 5))

    df.boxplot(column="url_count", by="label")

    plt.suptitle("")
    plt.title("Number of URLs in messages")
    plt.xlabel("Message type")
    plt.ylabel("Number of URLs")
    plt.show()


def plot_currency_count(df):
    """Show the number of currency symbols for ham and spam."""

    plt.figure(figsize=(8, 5))

    df.boxplot(column="currency_count", by="label")

    plt.suptitle("")
    plt.title("Number of currency symbols in messages")
    plt.xlabel("Message type")
    plt.ylabel("Number of currency symbols")
    plt.show()


def plot_number_count(df):
    """Show the number of digits for ham and spam."""

    plt.figure(figsize=(8, 5))

    df.boxplot(column="number_count", by="label")

    plt.suptitle("")
    plt.title("Number of digits in messages")
    plt.xlabel("Message type")
    plt.ylabel("Number of digits")
    plt.show()


def plot_class_distribution(df):
    """Show the class distribution in the dataset."""

    plt.figure(figsize=(6, 4))

    df["label"].value_counts().plot(
        kind="bar",
        color=["#2ca02c", "#d62728"]
    )

    plt.title("Class distribution in the dataset (Ham - 0 vs Spam - 1)")
    plt.xlabel("Class")
    plt.ylabel("Number of messages")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()
