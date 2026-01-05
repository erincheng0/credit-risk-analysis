import pandas as pd

# --------- loan default ---------


def engineer_features_loan_default(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "age" in df.columns:
        df["age_band"] = pd.cut(
            df["age"],
            bins=[17, 25, 35, 45, 55, 65, 120],
            labels=["18–25", "26–35", "36–45", "46–55", "56–65", "65+"]
        )

    if "income" in df.columns:
        df["income_band"] = pd.qcut(
            df["income"],
            q=4,
            labels=["Low", "Med-Low", "Med-High", "High"]
        )

    if "loan_amount" in df.columns:
        df["loan_amt_band"] = pd.qcut(
            df["loan_amount"],
            q=4,
            labels=["Small", "Med-Small", "Med-Large", "Large"]
        )

    if "dti_ratio" in df.columns:
        df["dti_bucket"] = pd.cut(
            df["dti_ratio"],
            bins=[0, 0.1, 0.2, 0.3, 0.4, 1.01],
            labels=["≤10%", "10–20%", "20–30%", "30–40%", ">40%"]
        )

    if "months_employed" in df.columns:
        df["emp_length_bucket"] = pd.cut(
            df["months_employed"],
            bins=[-1, 12, 36, 60, 120, 600],
            labels=["<1yr", "1–3yrs", "3–5yrs", "5–10yrs", "10+yrs"]
        )

    if "num_credit_lines" in df.columns:
        df["many_credit_lines"] = (df["num_credit_lines"] >= 6).astype(int)

    if {"dti_ratio", "loan_amount"}.issubset(df.columns):
        df["high_dti_large_loan"] = (
            (df["dti_ratio"] > 0.3) &
            (df["loan_amount"] > df["loan_amount"].median())
        ).astype(int)

    for col in ["has_mortgage", "has_dependents", "has_co_signer"]:
        if col in df.columns:
            df[col + "_Flag"] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    return df

# ------------- credit risk --------------


def engineer_features_credit_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "person_age" in df.columns:
        df["age_band"] = pd.cut(
            df["person_age"],
            bins=[17, 25, 35, 45, 55, 65, 120],
            labels=["18–25", "26–35", "36–45", "46–55", "56–65", "65+"]
        )

    if "person_income" in df.columns:
        df["income_band"] = pd.qcut(
            df["person_income"],
            q=4,
            labels=["Low", "Med-Low", "Med-High", "High"]
        )

    if "loan_amnt" in df.columns:
        df["loan_amnt_band"] = pd.qcut(
            df["loan_amnt"],
            q=4,
            labels=["Small", "Med-Small", "Med-Large", "Large"]
        )

    if "loan_percent_income" in df.columns:
        df["lpi_bucket"] = pd.cut(
            df["loan_percent_income"],
            bins=[0, 0.1, 0.2, 0.3, 0.4, 1.01],
            labels=["≤10%", "10–20%", "20–30%", "30–40%", ">40%"]
        )

    if "person_emp_length" in df.columns:
        df["emp_length_bucket"] = pd.cut(
            df["person_emp_length"],
            bins=[-1, 1, 3, 5, 10, 50],
            labels=["<1yr", "1–3yrs", "3–5yrs", "5–10yrs", "10+yrs"]
        )

    if "cb_person_default_on_file" in df.columns:
        df["prev_default_flag"] = df["cb_person_default_on_file"].map(
            {"Y": 1, "N": 0}
        ).astype(int)

    if {"loan_percent_income", "loan_grade"}.issubset(df.columns):
        low_grades = ["D", "E", "F", "G"]
        df["high_lpi_low_grade"] = (
            (df["loan_percent_income"] > 0.3) &
            (df["loan_grade"].isin(low_grades))
        ).astype(int)

    return df