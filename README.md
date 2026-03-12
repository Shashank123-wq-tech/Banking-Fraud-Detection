# Banking-Fraud-Detection
Machine learning–based system to detect fraudulent banking transactions by analyzing transaction patterns and identifying suspicious activities.

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
![Correlation Heatmap](Screenshot 2026-03-12 155546.png)

## Machine Learning Models
- Stochastic Gradient Descent
- XgboostClassifier

## Model Comparison

| Model | Accuracy | ROC-AUC-Score | Recall | PR-AUC|
|------|------|------|------|------|
|Stochastic Gradient Descent| 53% | 88% | 99% | 2% |
| XgboostClassifier | 99% | 98% | 98% | 84% |

The Random Forest model performed the best among all models and was selected as the final model.

## Some More Visualization of Model While Evaluating Xgboost.
![Correlation Heatmap]()
