# Credit Risk Analysis: Loan Default Prediction

## 1. Project Overview

A loan is a credit agreement where a lender provides money upfront and the borrower repays it over time with interest through scheduled payments. Default happens when the borrower can no longer make those payments, often because of income shocks, unexpected expenses, or taking on more debt than their finances can support. Understanding and predicting default risk is critical for lenders.

This project analyzes consumer credit data to understand why some borrowers default on their loans and to build a predictive model that flags high-risk applications before a lending decision is made. The business goal is to reduce losses from underperforming loans, improve approval decisions, and support more accurate risk-based pricing by estimating the probability that a borrower will fail to repay on time. To do this, the project cleans and engineers loan and borrower features, explores patterns linked to default, and trains classification models to estimate the likelihood of default for each loan.

The intended users are:
- The bank’s credit risk team, to validate and improve scorecards
- The lending product team, to refine underwriting rules and cut-offs
- Portfolio managers, to monitor borrower-level risk and adjust exposure proactively

---

## 2. Business Problem

This project is framed from the perspective of a consumer lender managing a large portfolio of unsecured personal loans. Each loan is relatively small, but the aggregate exposure is significant for profitability and regulatory capital. The portfolio already contains loans that are severely delinquent or written off, and the default rate both creates credit losses and highlights which customer segments—by income, debt-to-income ratio, job stability, or past delinquencies—are riskier.

The core business questions are:

- What is the current default rate, and which customer or product segments contribute most to it?
- Which borrower and loan characteristics (such as income, employment length, debt-to-income ratio, loan amount, term, and interest rate) are most strongly associated with default?
- How accurately can default risk be predicted for new applicants using historical performance data?

Success means building a model and policy framework that reduces default and expected loss while keeping approval rates at or above a target level, and keeping risk metrics—such as default rate, loss rate, and risk-weighted assets—within the lender’s risk appetite and regulatory requirements.

---

## 3. Data

This project uses two open-source Kaggle datasets:

1. [**Loan Default Prediction**](https://www.kaggle.com/datasets/nikhil1e9/loan-default) by Nikhil  
   - Contains individual loan applications and outcomes for a bank loan default prediction challenge

2. [**Credit Risk Dataset**](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data) by Lao Tse  
   - Contains credit-bureau-style attributes and loan performance information for consumer borrowers

Together, the datasets include:
- Borrower demographics such as age, income, and employment details
- Loan features such as loan amount, interest rate, term, and purpose
- A target label indicating whether each loan defaulted

The project uses the raw CSV files from Kaggle. The key columns capture borrower characteristics, loan attributes, and a binary target variable indicating default vs. non-default, where “default” follows each dataset’s original definition of serious delinquency or charge-off.

### Data limitations
- Some variables are anonymized or simulated
- Time coverage is not fully specified
- The target class is imbalanced
- The data excludes some real-world policy and behavioral variables

### Why the datasets are analyzed separately
The two datasets are not merged because they come from different sources with different schemas and feature definitions. Merging them would mix heterogeneous populations and label definitions, likely introducing dataset shift and making model results harder to interpret. Instead, each dataset is treated as its own experiment, and results are compared only at a high level.

---

## 4. Methodology

### 4.1 Data Cleaning and Preprocessing

A consistent preprocessing pipeline is applied to both datasets before modeling.

#### Column standardization and duplicates
- All column names are converted to `snake_case`
- Obvious naming issues are fixed (for example, `loan_i_d` → `loan_id`, `d_t_i_ratio` → `dti_ratio`)
- Duplicate records are removed

#### Handling missing values and outliers
- Numeric fields are coerced to numeric types
- Invalid values are treated as missing
- In the Credit Risk dataset, missing values in fields such as `person_emp_length` and `loan_int_rate` are imputed with the median
- In the Loan Default dataset, extreme values in `dti_ratio` and `interest_rate` are winsorized at the 1st and 99th percentiles

#### Encoding categorical variables
Categorical features are cast to categorical types to prepare them for downstream encoding such as one-hot encoding.

#### Target variable definition
- In the Loan Default dataset, `default` is converted to an integer binary flag
- In the Credit Risk dataset, `status` is renamed to `default_flag` and converted to an integer binary flag

#### Train / validation / test split
Each dataset is split into training, validation, and test sets to support model tuning and unbiased final evaluation.

---

### 4.2 Feature Engineering

#### Loan Default dataset

- **Age bands (`age_band`)**  
  Ages are grouped into bands (18–25, 26–35, 36–45, 46–55, 56–65, 65+) to capture non-linear age effects.

- **Income and loan amount bands (`income_band`, `loan_amt_band`)**  
  Income and loan amount are split into quartile-based groups for easier segment comparison.

- **Debt-burden buckets (`dti_bucket`)**  
  `dti_ratio` is grouped into intuitive affordability buckets:
  - ≤10%
  - 10–20%
  - 20–30%
  - 30–40%
  - >40%

- **Employment length and credit lines (`emp_length_bucket`, `many_credit_lines`)**  
  Employment duration is bucketed by tenure, and a binary flag identifies borrowers with six or more credit lines.

- **High DTI × large loan interaction (`high_dti_large_loan`)**  
  Flags borrowers with both high DTI and above-median loan amount.

- **Household and structure flags (`has_mortgage_flag`, etc.)**  
  Yes/No variables for mortgage, dependents, and co-signer are converted into binary indicators.

#### Credit Risk dataset

- **Age, income, and loan amount bands (`age_band`, `income_band`, `loan_amnt_band`)**  
  Similar segment bands are created for consistency across datasets.

- **Loan payment-to-income buckets (`lpi_bucket`)**  
  `loan_percent_income` is grouped into affordability buckets.

- **Employment length buckets (`emp_length_bucket`)**  
  Employment duration is bucketed into interpretable groups.

- **Previous default flag (`prev_default_flag`)**  
  `cb_person_default_on_file` is mapped from Y/N to 1/0.

- **High payment burden × low grade interaction (`high_lpi_low_grade`)**  
  Flags borrowers with both high payment burden and lower credit grades (D–G).

---

### 4.3 Exploratory Data Analysis (EDA)

EDA is used to understand how key borrower and loan features relate to default in each dataset. The analysis begins with:
- Data types
- Descriptive statistics
- Target class balance
- Missing value checks

The visual analysis then focuses on default patterns across important features.

#### Credit Risk dataset insights
- Borrowers with higher `loan_percent_income` show much higher default rates
- The distribution of `loan_percent_income` for defaulters is shifted toward higher values
- Higher-risk loan grades (such as F and G) have much higher default rates than lower-risk grades

#### Loan Default dataset insights
- Younger borrowers (18–25 and 26–35) have the highest default rates
- Default risk tends to rise with higher DTI ratios
- DTI-related patterns are visible in both continuous and bucketed views

These EDA findings help identify which features are likely to be useful in modeling.

---

### 4.4 Modeling Approach

This project frames credit risk as a supervised binary classification problem: predict whether a loan will default (1) or not (0) based on borrower and loan features.

Each dataset is split using a two-stage stratified split so that the default rate is preserved in train, validation, and test sets. Numerical features are standardized using a scaler fitted only on the training data to avoid information leakage.

Three models are trained on each dataset:

- **Logistic Regression**  
  Used as a linear, interpretable baseline

- **Random Forest Classifier**  
  Used to capture non-linear relationships and feature interactions

- **Gradient Boosting Classifier**  
  Used to improve predictive ranking performance on structured tabular data

Class weights are set to `"balanced"` for logistic regression and random forest to partially address class imbalance.

### Evaluation metrics
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 score
- ROC AUC
- Confusion matrices

For the `credit_risk` dataset, gradient boosting is selected as the primary champion model. For the `loan_default` dataset, results show a trade-off between higher recall from logistic regression and more conservative predictions from tree-based models.

---

## 5. Results

### 5.1 Model Performance

#### Loan Default dataset
- Logistic regression achieves test ROC AUC of about **0.75**
- Recall is relatively high at about **0.70**
- Precision is low at about **0.22**

This means the model catches many true defaulters but also flags many non-defaulters as risky.

Gradient boosting:
- Slightly improves ROC AUC to about **0.76**
- Achieves overall accuracy around **0.89**
- Has much lower recall at the default 0.5 threshold
- Has higher precision around **0.58**

This means gradient boosting behaves more conservatively and misses more defaulters unless the threshold is adjusted.

#### Credit Risk dataset
- Baseline logistic regression already performs well with test ROC AUC of about **0.83**
- Gradient boosting improves performance further and is selected as the final model
- Final gradient boosting results:
  - ROC AUC ≈ **0.886**
  - Precision ≈ **0.76**
  - Recall ≈ **0.60**

This model provides a strong balance between identifying defaulters and avoiding unnecessary rejection of low-risk applicants.

---

### 5.2 Key Risk Drivers

Model explainability tools are used to identify the strongest predictors of default in each dataset.

#### Credit Risk dataset
The most important drivers are:
- Higher `loan_percent_income`
- Higher `loan_int_rate`
- Lower `person_income`

Secondary factors include:
- Employment length
- Credit history length
- Loan amount

#### Loan Default dataset
The most important drivers are:
- Number of existing credit lines
- Months employed
- Income
- Credit score
- Loan amount

Higher risk is associated with:
- More credit lines
- Shorter employment history
- Higher DTI
- Longer loan terms

Lower risk is associated with:
- Higher credit score
- Higher income

#### Example borrower profiles

**High-risk profile**
- Younger borrower
- Short employment history
- Several active credit lines
- Lower income
- Large loan
- High interest rate
- High payment-to-income ratio

**Low-risk profile**
- Older borrower
- Long and stable employment history
- Few credit lines
- Strong credit score
- Higher income
- Modest loan amount
- Low payment-to-income ratio

---

### 5.3 Risk Bands and Portfolio View

Predicted default probabilities (PD) are grouped into three risk bands:

- **Low risk:** PD < 5%
- **Medium risk:** 5% ≤ PD < 15%
- **High risk:** PD ≥ 15%

Each test portfolio is scored using the gradient boosting model and summarized with:

- `pd_hat`: predicted probability of default
- `risk_band`: assigned risk band
- `exposure`: loan amount treated as exposure
- `expected_loss`: `PD × exposure × LGD`

In this project, LGD (loss given default) is assumed to be **45%**.

Portfolio summaries report:
- Number of loans in each band
- Share of total exposure
- Observed default rate
- Average predicted PD
- Total expected loss

In both datasets, higher risk bands show materially higher observed default rates and expected losses, indicating meaningful concentrations of credit risk.

---

## 6. Business Recommendations

Based on the model results and risk-band analysis, several practical actions are possible.

### Risk-based policy actions

- **Low risk band**
  - Maintain streamlined approval
  - Offer competitive pricing

- **Medium risk band**
  - Require additional documentation
  - Consider smaller maximum loan amounts
  - Consider shorter loan terms

- **High risk band**
  - Tighten approval criteria
  - Cap loan size and tenor
  - Selectively decline or require collateral

### Risk-based pricing
Because expected loss rises sharply across PD bands, lenders can:
- Increase interest rates or fees for medium- and high-risk borrowers
- Reserve the most attractive offers for low-risk borrowers

### Underwriting workflow integration
PD estimates and risk bands can be embedded directly into underwriting rules:

- **Low risk:** auto-approve if basic eligibility checks pass
- **Medium risk:** manual review with focus on DTI, employment length, and number of credit lines
- **High risk:** approve only with mitigants or decline

### Portfolio monitoring
Risk bands can also support ongoing portfolio monitoring:
- Track exposure by band over time
- Monitor borrowers whose PD rises from low → medium or medium → high
- Use these shifts as early-warning signals for outreach, limit reductions, or restructuring

### Ethics and fairness
Credit-risk models should rely only on economically meaningful variables such as income, payment burden, interest rate, credit score, employment length, and number of credit lines. Variables that directly encode or closely proxy protected characteristics should not be used.

In a production setting, fairness metrics should be monitored regularly, including:
- Approval rate differences across groups
- Default rate differences across groups
- Pricing outcome differences across groups

### Model risk and maintenance
The PD estimates in this project are learned from a single static snapshot of data. In a real production environment, model performance could degrade as economic conditions, underwriting standards, or product design change.

A lender would therefore need:
- Regular recalibration
- Backtesting
- Performance monitoring
- Periodic model redevelopment

---

## 7. Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 8. Repository Structure

```bash
.
├── data/
├── notebooks/
├── src/
├── outputs/
└── README.md
```

---

## 9. Key Takeaways

- Default risk can be modeled effectively as a binary classification problem
- Payment burden, interest rate, income, employment stability, and credit usage are important predictors of default
- Gradient boosting performs best on the `credit_risk` dataset
- The `loan_default` dataset highlights the trade-off between aggressive default detection and conservative approval strategy
- Risk bands make model outputs easier to use in pricing, underwriting, and portfolio monitoring
