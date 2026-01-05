

# src/interpretation.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_logistic_coefficients(model, feature_cols, title_prefix: str) -> None:
    coef = model.coef_[0]

    if len(coef) != len(feature_cols):
        n = min(len(coef), len(feature_cols))
        coef = coef[:n]
        feature_cols = feature_cols[:n]

    coef_df = (
        pd.DataFrame({"feature": feature_cols, "coefficient": coef})
        .sort_values("coefficient", ascending=False)
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=coef_df,
        x="coefficient",
        y="feature",
        palette="coolwarm",
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.title(f"{title_prefix} – Logistic Regression Coefficients")
    plt.xlabel("Coefficient (log-odds impact on default)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()




def plot_feature_importances(model, feature_cols, title_prefix: str, model_name: str) -> None:
    importances = model.feature_importances_

    # Align lengths if they differ
    if len(importances) != len(feature_cols):
        n = min(len(importances), len(feature_cols))
        importances = importances[:n]
        feature_cols = feature_cols[:n]

    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=True)
    )

    plt.figure(figsize=(8, 6))
    plt.barh(imp_df["feature"], imp_df["importance"])
    plt.xlabel("Importance")
    plt.title(f"{title_prefix} – {model_name} Feature Importances")
    plt.tight_layout()
    plt.show()

