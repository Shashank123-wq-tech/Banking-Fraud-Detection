# Financial Fraud Detection System.

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange?logo=xgboost&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-3.x-9146FF)
![Pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

![GitHub stars](https://img.shields.io/github/stars/Shashank123-wq-tech/Banking-Fraud-Detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/Shashank123-wq-tech/Banking-Fraud-Detection?style=social)
![Last commit](https://img.shields.io/github/last-commit/Shashank123-wq-tech/Banking-Fraud-Detection)
![Repo size](https://img.shields.io/github/repo-size/Shashank123-wq-tech/Banking-Fraud-Detection)

Machine learning system for detecting account-takeover fraud in mobile money
transactions, built on the Financial simulated transaction dataset. Trained and
compared four classifiers — SGD, Logistic Regression, LightGBM, and XGBoost —
to flag fraudulent TRANSFER and CASH_OUT transactions in real time, with the
long-term goal of evolving this into a full production fraud detection
platform.

## Problem Statement

Mobile money platforms are vulnerable to account-takeover fraud, where an
attacker gains unauthorized control of a customer's account and attempts to
drain it before the legitimate owner notices. In the PaySim simulation, this
fraud pattern follows a consistent two-step chain: a **TRANSFER** that fully
empties the victim's account balance, followed by a **CASH_OUT** that extracts
the funds from the system via a mule account.

Empirical analysis of the full dataset (6,362,620 transactions) confirms fraud
is structurally confined to these two transaction types:

| Type      | Transactions | Fraud cases | Fraud rate |
|-----------|-------------:|------------:|-----------:|
| CASH_IN   | 1,399,284    | 0           | 0.00%      |
| DEBIT     | 41,432       | 0           | 0.00%      |
| PAYMENT   | 2,151,495    | 0           | 0.00%      |
| TRANSFER  | 532,909      | 4,097       | 0.77%      |
| CASH_OUT  | 2,237,500    | 4,116       | 0.18%      |

No fraud occurs in CASH_IN, DEBIT, or PAYMENT transactions, and all destination
accounts in the TRANSFER/CASH_OUT population are verified customer accounts —
the merchant-related data quality artifacts present in the full dataset
(untracked balances) do not affect this modeling population. The existing
rule-based control (`isFlaggedFraud`, which flags single transfers over
200,000) fails to catch the vast majority of fraud, which frequently occurs at
much smaller amounts.

## Goal

Build a binary classification model that scores TRANSFER and CASH_OUT
transactions with the probability that they represent account-takeover fraud,
enabling interception before funds exit the system — while minimizing false
positives that would disrupt legitimate customers.

**Success criteria:**
- Maximize recall (fraud catch rate) at an acceptable precision threshold,
  given the cost asymmetry between missed fraud and false holds
- Primary metric: Precision-Recall AUC (not accuracy, given ~0.3% fraud
  prevalence in the scoped population)
- Report performance in business terms: *"At X% precision, the model catches
  Y% of fraudulent transactions"*

## Approach

1. **Scoping** — filtered to TRANSFER/CASH_OUT only, verified empirically
   rather than assumed (see table above)
2. **EDA** — profiled balance-conservation behavior; identified that fraud
   correlates with an account being fully drained (`oldbalanceOrg == amount`,
   `newbalanceOrig == 0`), not with transaction size alone
3. **Feature engineering** — log-transforms, behavioral ratios, and
   balance-conservation checks; full breakdown in
   [Feature Engineering](#feature-engineering) below
4. **Train/test split** — time-based split using `step`, rather than random,
   to avoid leakage and simulate real deployment conditions
5. **Modeling** — trained and compared SGD, Logistic Regression, LightGBM, and
   XGBoost, with class-imbalance handling evaluated via precision-recall
   metrics rather than accuracy alone

## Feature Engineering

### Transformations

- `amount` → `log_amount`
- `oldbalanceOrg` → `log_oldbalanceOrg`

Log-transforms reduce the heavy right-skew typical of monetary values, making
these features better behaved for models sensitive to scale.

### Behavioral Features

| Feature | Formula | Purpose |
|---|---|---|
| `sender_balance_change` | `oldbalanceOrg − newbalanceOrig` | Amount that actually left the origin account |
| `receiver_balance_change` | `newbalanceDest − oldbalanceDest` | Amount that actually arrived at the destination |
| `amount_sender_ratio` | `amount / (oldbalanceOrg + 1)` | Fraction of the sender's balance moved in this transaction |
| `sender_error` | `oldbalanceOrg − amount − newbalanceOrig` | Conservation-of-money check on the origin side (should be ≈0) |
| `receiver_error` | `oldbalanceDest + amount − newbalanceDest` | Conservation-of-money check on the destination side (should be ≈0) |
| `emptied_account` | `1 if newbalanceOrig == 0 else 0` | Flags accounts fully drained by the transaction |
| `dest_balance_missing` | `1 if oldbalanceDest == 0 and newbalanceDest == 0 else 0` | Flags destination accounts with no tracked balance |

**`amount_sender_ratio` interpretation:**
- ≈ 0.05 → normal transaction
- ≈ 0.95 → account nearly emptied
- \> 1 → suspicious / inconsistent

### Feature Selection

- **Correlation-based pruning** — removed features with pairwise correlation
  above 0.90 to reduce redundancy before modeling
- **Random Forest feature importance** — used to validate that engineered
  features carry real signal rather than being included on intuition alone

| Feature | Importance |
|---|---:|
| `newbalanceOrig` | 0.1454 |
| `amount_sender_ratio` | 0.1394 |
| `dest_balance_missing` | 0.1333 |
| `emptied_account` | 0.1098 |
| `sender_balance_change` | 0.1086 |
| `type_CASH_OUT` | 0.0454 |
| `type_TRANSFER` | 0.0453 |
| `sender_error` | 0.0433 |
| `oldbalanceOrg` | 0.0389 |
| `log_oldbalanceOrg` | 0.0342 |
| `newbalanceDest` | 0.0319 |
| `receiver_balance_change` | 0.0307 |
| `step` | 0.0224 |
| `receiver_error` | 0.0192 |
| `amount` | 0.0175 |
| `oldbalanceDest` | 0.0174 |
| `log_amount` | 0.0173 |
| `isFlaggedFraud` | 0.0002 |

Five of the top six features by importance are engineered, not raw —
validating the feature engineering approach. Notably, `isFlaggedFraud` (the
existing rule-based control) ranks last by a wide margin, reinforcing the
problem statement's finding that the current 200,000-threshold rule
contributes almost nothing to fraud detection.

## Model Comparison

| Model                        | Accuracy | Precision | Recall  | F1 Score | ROC-AUC | PR-AUC  |
|-------------------------------|---------:|----------:|--------:|---------:|--------:|--------:|
| Stochastic Gradient Descent   | 97%      | —         | 86%     | —        | 0.92    | 0.04    |
| Logistic Regression            | 96.96%   | 8.79%     | 98.54%  | 0.1613   | 0.9963  | —       |
| LightGBM                       | 98.05%   | 12.93%    | 97.44%  | 0.2284   | 0.9775  | 0.1261  |
| **XGBoost (selected)**         | **99.99%** | **98.26%** | **99.70%** | **0.9897** | **0.9984** | **0.9979** |

**Why XGBoost was selected:** Every baseline — SGD, Logistic Regression, and
LightGBM — reaches respectable accuracy (96–98%) and, on paper, workable
recall (86–98%). PR-AUC is what exposes the real gap: SGD collapses to 0.04
and LightGBM to 0.1261, both far below XGBoost's 0.9979. This is exactly why
**PR-AUC, not accuracy or ROC-AUC, is the metric that matters under severe
class imbalance** — SGD's ROC-AUC (0.92) looks passable in isolation, but a
PR-AUC of 0.04 means it is effectively unusable at any reasonable precision
threshold, since nearly every transaction it flags as fraud would be a false
alarm. XGBoost is the only model that holds precision and recall together
(98.26% / 99.70%), which is why it was selected as the final model.

## Model Evaluation

### Confusion Matrices

<table>
<tr>
<td align="center"><b>Logistic Regression</b><br><img src="assets/confusion_matrix_logreg.png" width="260"/></td>
<td align="center"><b>LightGBM</b><br><img src="assets/confusion_matrix_lgbm.png" width="260"/></td>
<td align="center"><b>XGBoost</b><br><img src="assets/confusion_matrix_xgboost.png" width="260"/></td>
</tr>
</table>

*(Export confusion matrix plots to `assets/` and update the paths above.)*

### Feature Correlation — Before vs After Feature Engineering

<table>
<tr>
<td align="center"><b>Before feature engineering</b><br><img src="assets/correlation_before.png" width="380"/></td>
<td align="center"><b>After feature engineering</b><br><img src="assets/correlation_after.png" width="380"/></td>
</tr>
</table>

*(Export correlation heatmaps to `assets/` and update the paths above. Useful
to show how engineered features like `sender_error` and `emptied_account`
correlate more strongly with `isFraud` than the raw balance columns did.)*

## Tech Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · Jupyter · Matplotlib / Seaborn

## Repository Structure

```
├── assets/                 # Plots for README (confusion matrices, correlation heatmaps)
├── data/                   # PaySim dataset (or download instructions)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── models/                 # Saved model artifacts
├── requirements.txt
└── README.md
```

## How to Run

```bash
git clone https://github.com/Shashank123-wq-tech/Banking-Fraud-Detection.git
cd Banking-Fraud-Detection
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb
```

## Roadmap — Toward a Full Fraud Detection Platform

The current repository covers model development and evaluation. The planned
next phase is to productionize this into a full system, targeting the
following architecture:

- **Serving** — FastAPI prediction service behind an API Gateway, with a
  Streamlit analyst dashboard for review and reporting
- **Explainability** — SHAP-based feature importance, waterfall/force plots
  surfaced alongside every prediction
- **Risk & decisioning** — a risk-scoring layer translating model probability
  into actions (approve / require OTP / manual review / block)
- **MLOps** — MLflow for experiment tracking, model versioning, and a model
  registry
- **Infrastructure** — Dockerized microservices deployed on AWS (ECS Fargate,
  RDS for prediction/audit history, S3 for artifacts), provisioned via
  Terraform
- **CI/CD** — GitHub Actions pipeline (lint, test, build, deploy) with health
  checks and smoke tests before production rollout
- **Security & auth** — JWT-based authentication with role-based access
  control (analyst / admin / auditor / developer roles), audit logging

This roadmap is aspirational and not yet implemented — it reflects the target
system design for future development, not the current state of the repo.

## Author

Shashank — [GitHub](https://github.com/Shashank123-wq-tech)


