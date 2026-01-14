
import numpy as np
import pandas as pd


def compute_pd(model, X):
    """Return predicted PD (probability of default) as a 1D array."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores).ravel()
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    else:
        raise ValueError("Model has neither predict_proba nor decision_function.")


def assign_risk_band(pd_series, bins=None, labels=None):
    """
    Assign PDs to risk bands.
    Default bands: [0–5%), [5–15%), [15–100%).
    """
    if bins is None:
        bins = [0.0, 0.05, 0.15, 1.0]
    if labels is None:
        labels = ["Low", "Medium", "High"]

    return pd.cut(
        pd_series,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )


def build_portfolio_df(X, y, model, exposure_col_name, lgd=0.45):
    """
    Build a portfolio DataFrame with PD, risk band, exposure, and expected loss.
    X: feature DataFrame used for scoring.
    y: true default labels (Series or array).
    exposure_col_name: column in X to treat as exposure (e.g. 'loan_amount').
    """
    X = X.copy()
    y = pd.Series(y).reset_index(drop=True)

    pd_hat = compute_pd(model, X)
    pd_hat = pd.Series(pd_hat, index=X.index, name="pd_hat")

    portfolio = X.copy()
    portfolio["default"] = y.values
    portfolio["pd_hat"] = pd_hat.values

    portfolio["risk_band"] = assign_risk_band(portfolio["pd_hat"])

    portfolio["exposure"] = portfolio[exposure_col_name]
    portfolio["expected_loss"] = portfolio["pd_hat"] * portfolio["exposure"] * lgd

    return portfolio


def summarize_by_band(portfolio_df):
    """
    Return a summary table by risk_band: volume, exposure, default_rate, avg_pd, expected_loss.
    """
    summary = (
        portfolio_df
        .groupby("risk_band")
        .agg(
            n_loans=("default", "size"),
            exposure_sum=("exposure", "sum"),
            default_rate=("default", "mean"),
            avg_pd=("pd_hat", "mean"),
            expected_loss_sum=("expected_loss", "sum"),
        )
        .reset_index()
    )

    total_exposure = summary["exposure_sum"].sum()
    if total_exposure > 0:
        summary["exposure_share"] = summary["exposure_sum"] / total_exposure
    else:
        summary["exposure_share"] = 0.0

    return summary
