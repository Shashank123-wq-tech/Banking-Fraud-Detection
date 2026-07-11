# Banking-Fraud-Detection
Machine learning–based system to detect fraudulent banking transactions by analyzing transaction patterns and identifying suspicious activities.

## Problem Statement

Mobile money platforms are vulnerable to account-takeover fraud, where an attacker
gains unauthorized control of a customer's account and attempts to drain it before
the legitimate owner notices. In the PaySim simulation, this fraud pattern follows
a consistent two-step chain: a **TRANSFER** that fully empties the victim's account
balance, followed by a **CASH_OUT** that extracts the funds from the system via a
mule account.

Empirical analysis of the full dataset (6,362,620 transactions) confirms that fraud
is structurally confined to these two transaction types:

| Type      | Transactions | Fraud cases | Fraud rate |
|-----------|-------------:|------------:|-----------:|
| CASH_IN   | 1,399,284    | 0           | 0.00%      |
| DEBIT     | 41,432       | 0           | 0.00%      |
| PAYMENT   | 2,151,495    | 0           | 0.00%      |
| TRANSFER  | 532,909      | 4,097       | 0.77%      |
| CASH_OUT  | 2,237,500    | 4,116       | 0.18%      |

No fraud occurs in CASH_IN, DEBIT, or PAYMENT transactions, and all destination
accounts in the TRANSFER/CASH_OUT population are verified customer accounts
(0 merchant destinations) — the merchant-related data quality artifacts present
in the full dataset (untracked balances) do not affect this modeling population.

Existing rule-based controls (`isFlaggedFraud`: flagging single transfers over
200,000) fail to catch the vast majority of fraud, which frequently occurs at
much smaller transaction amounts. This leaves a critical detection gap that a
supervised machine learning model can address.

## Goal

Build a binary classification model that scores TRANSFER and CASH_OUT
transactions in real time with the probability that they represent
account-takeover fraud, enabling the platform to intercept fraudulent transfers
and cash-outs before funds exit the system — while minimizing false positives
that would disrupt legitimate customers.

**Success criteria:**
- Maximize **recall** (fraud catch rate) at an acceptable **precision** threshold,
  given the severe cost asymmetry between missed fraud (direct financial loss)
  and false holds (customer friction)
- Primary evaluation metric: **Precision-Recall AUC** (not accuracy or ROC-AUC
  alone, given ~0.3% fraud prevalence in the scoped population)
- Report business-facing performance as: *"At X% precision, the model catches
  Y% of fraudulent transactions"*

# Project Objectives
- Detect fraudulent transactions.
- Reduce financial losses.
- Build a machine learning classification model.
- Evaluate model performance.

 # Dataset Information
 There are 6362620 rows and 11 columns in this dataset with description of each column given below:
 step - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).

type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.

amount - amount of the transaction in local currency.

nameOrig - customer who started the transaction

oldbalanceOrg - initial balance before the transaction

newbalanceOrig - new balance after the transaction

nameDest - customer who is the recipient of the transaction

oldbalanceDest - initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).

newbalanceDest - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).

isFraud - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.

isFlaggedFraud - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

  ## Project Workflow
1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training
6. Model Evaluation

## Visualization :
The project includes several visualizations to understand the dataset and fraud patterns.
### Correlation Heatmap
![Correlation Heatmap](https://github.com/Shashank123-wq-tech/Banking-Fraud-Detection/blob/main/Screenshot%202026-04-20%20160251.png)

## Machine Learning Models
- Stochastic Gradient Descent
- XgboostClassifier

## Model Comparison

| Model | Accuracy | ROC-AUC-Score | Recall | PR-AUC|
|------|------|------|------|------|
|Stochastic Gradient Descent| 97% | 92% | 86% | 4% |
| XGBClassifier | 99% | 99% | 99% | 96% |

The Random Forest model performed the best among all models and was selected as the final model.

### Some More Visualization of Model While Evaluating Randan Forest.
![Precision - Recall Curve](https://github.com/Shashank123-wq-tech/Banking-Fraud-Detection/blob/main/Screenshot%202026-04-20%20160310.png)

### Confusion Matrix
![Confusion Matrix](https://github.com/Shashank123-wq-tech/Banking-Fraud-Detection/blob/main/Screenshot%202026-04-20%20160322.png)
### Interpretation:
1) A recall of 99% means the model detects almost all actual fraud cases, missing very few. This is critical because missing fraud is usually more costly than false alarms.
 
2) Precision-Recall AUC is especially important for imbalanced datasets. A score of 96% shows that the model maintains a very strong balance between:
(i) catching fraud (recall).
(ii) avoiding false positives (precision).

3) This is extremely strong. A ROC-AUC of 0.99 means the model can almost perfectly distinguish between fraud and non-fraud cases.
 
4) In real-world fraud detection, the XGBClassifier effectively identifies almost all fraudulent transactions with high precision, minimizing both missed frauds and false alarms.
