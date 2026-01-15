
import pandas as pd

def to_snake_case(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(r"(?<!^)(?=[A-Z])", "_", regex=True)
        .str.lower()
    )
   
    return df

# ---------- Loan default ----------
def clean_loan_default(df: pd.DataFrame) -> pd.DataFrame:
    # make sure this is a real copy, not tied to any slice
    df = df.copy()
    print("Inside clean_loan_default, initial cols:", df.columns.tolist())

    # 1) Fix the two weird names FIRST
    df = df.rename(columns={
        "loan_i_d": "loan_id",
        "d_t_i_ratio": "dti_ratio",
    })

    print("After rename:", df.columns.tolist())

    # 2) Drop duplicates using the new name
    df = df.drop_duplicates(subset=["loan_id"])

    # 3) Cast types
    cat_cols = [
        "education", "employment_type", "marital_status",
        "has_mortgage", "has_dependents", "loan_purpose", "has_co_signer",
    ]
    for col in cat_cols:
        df.loc[:, col] = df[col].astype("category")

    num_cols = [
        "age", "income", "loan_amount", "credit_score",
        "months_employed", "num_credit_lines",
        "interest_rate", "loan_term", "dti_ratio",
    ]
    for col in num_cols:
        df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[:, "default"] = df["default"].astype(int)

    for col in ["dti_ratio", "interest_rate"]:
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        df.loc[:, col] = df[col].clip(lower=q1, upper=q99)

    return df



# ---------- Credit risk ----------

def clean_credit_risk(df: pd.DataFrame) -> pd.DataFrame:
    # make a real copy once
    df = df.copy()

    df = df.drop_duplicates()

    cat_cols = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]
    for col in cat_cols:
        if col in df.columns:
            df.loc[:, col] = df[col].astype("category")

    num_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]
    for col in num_cols:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")

    if "person_emp_length" in df.columns:
        df.loc[:, "person_emp_length"] = df["person_emp_length"].fillna(
            df["person_emp_length"].median()
        )

    if "loan_int_rate" in df.columns:
        df.loc[:, "loan_int_rate"] = df["loan_int_rate"].fillna(
            df["loan_int_rate"].median()
        )

    if "loan_status" in df.columns:
        df = df.rename(columns={"loan_status": "default_flag"})

    df.loc[:, "default_flag"] = df["default_flag"].astype(int)

    return df




