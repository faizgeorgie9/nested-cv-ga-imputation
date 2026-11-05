# Benchmark Dataset: BankChurners

This directory contains the benchmark credit card customer dataset utilized in the study:

- **Filename**: `BankChurners.csv`
- **Source**: Kaggle Credit Card Customers Dataset ([sakshigoyal7/credit-card-customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers))
- **Total Records ($N$)**: 10,127 customer instances
- **Total Attributes ($D$)**: 19 predictive features (13 continuous and 6 categorical)
- **Target Variable**: `Attrition_Flag` (Binary: 16.07% attrited/churners vs. 83.93% existing customers)

### Feature Overview
- **Continuous Features (13)**: `Customer_Age`, `Dependent_count`, `Months_on_book`, `Total_Relationship_Count`, `Months_Inactive_12_mon`, `Contacts_Count_12_mon`, `Credit_Limit`, `Total_Revolving_Bal`, `Avg_Open_To_Buy`, `Total_Amt_Chng_Q4_Q1`, `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`.
- **Categorical Features (6)**: `Gender`, `Education_Level`, `Marital_Status`, `Income_Category`, `Card_Category`.
- **Excluded Features**: `CLIENTNUM` (identifier) and two extraneous naive Bayes classifier benchmark artifacts columns (`Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1` and `_2`).

### Preprocessing & Quarantine Barrier
All categorical features are integer-encoded, and continuous features are standardized using parameters $(\mu_j, \sigma_j)$ computed strictly from the training partition in each fold. Under the zero-leakage nested cross-validation protocol, outer validation partitions are strictly quarantined and never accessed during preprocessing or imputation fitting.
