import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def basic_summary(df: pd.DataFrame, target: str | None = None) -> None:
    print(df.info())
    print(df.describe(include="all"))
    if target is not None and target in df.columns:
        print("\nTarget distribution:")
        print(df[target].value_counts(normalize=True))


def plot_default_rate(df: pd.DataFrame, col: str, target: str, title: str | None = None, rotation: int = 45) -> None:
    tmp = (
        df[[col, target]]
        .groupby(col, observed=False)
        .mean()
        .reset_index()
        .sort_values(target, ascending=False)
    )
    plt.figure(figsize=(7, 4))
    sns.barplot(data=tmp, x=col, y=target)
    plt.ylabel("Default rate")
    if title:
        plt.title(title)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.show()


def kde_by_target(df: pd.DataFrame, col: str, target: str) -> None:
    plt.figure(figsize=(7, 4))
    sns.kdeplot(data=df, x=col, hue=target, common_norm=False)
    plt.title(f"{col} distribution by {target}")
    plt.tight_layout()
    plt.show()
