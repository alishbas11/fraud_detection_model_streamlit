# Fraud Detection Notebook

## Introduction
This notebook presents a comprehensive machine learning pipeline designed to detect fraudulent transactions within a financial dataset. The process encompasses data loading, extensive exploratory data analysis (EDA), data preprocessing, model training, and evaluation, ultimately aiming to build a robust fraud detection system.

## Dataset Description
The dataset contains transactional information, including `step` (mapping to a unit of time in the real world), `type` of transaction (e.g., PAYMENT, TRANSFER, CASH_OUT), `amount` of transaction, original and new balances of both origin (`nameOrig`) and destination (`nameDest`) accounts, and flags indicating actual fraud (`isFraud`) and system-flagged fraud (`isFlaggedFraud`).

## Tools and Their Use

### 1. Data Loading and Initial Inspection
*   **`pandas`**: Used for reading the CSV file (`pd.read_csv`), displaying the first few rows (`df.head()`), examining column names (`df.columns`), and getting a summary of the DataFrame's structure, data types, and non-null values (`df.info()`).

### 2. Exploratory Data Analysis (EDA)
*   **`pandas`**: Employed for various data manipulations:
    *   Filtering data based on conditions (`df.loc[]`).
    *   Creating cross-tabulations (`pd.crosstab()`) to understand the relationship between `isFraud` and `isFlaggedFraud`.
    *   Counting unique values in columns (`df.value_counts()`) for categorical distribution.
    *   Grouping data and aggregating fraud counts by transaction type (`df.groupby()`).
*   **`matplotlib.pyplot` (`plt`) & `seaborn` (`sns`)**: Utilized for data visualization:
    *   A heatmap (`sns.heatmap()`) is generated to visualize correlations between numerical features.
    *   Box plots (`sns.boxplot()`) illustrate the distribution of transaction amounts with respect to fraud status.
    *   Histograms (`sns.histplot()`) show the distribution of log-transformed transaction amounts to handle skewness.
    *   Count plots (`sns.countplot()`) display the distribution of transaction types, differentiating between fraudulent and non-fraudulent transactions.

### 3. Data Preprocessing & Feature Engineering
*   **`pandas`**: A new feature, `balancediffDest`, is created to capture the difference between `newbalanceDest` and `oldbalanceDest`, which might be indicative of fraudulent activity.
*   Irrelevant columns such as `step`, `isFlaggedFraud`, `nameOrig`, and `nameDest` are dropped from the dataset (`df_model`) before model training.

### 4. Model Training
*   **`scikit-learn`**: A robust machine learning library is used for building the model:
    *   `StandardScaler`: Scales numerical features to have zero mean and unit variance.
    *   `OneHotEncoder`: Converts the categorical 'type' feature into a numerical one-hot encoded format.
    *   `ColumnTransformer`: Applies different transformations to different columns (numerical columns get scaling, categorical columns get one-hot encoding).
    *   `Pipeline`: Chains together the preprocessing steps and the classifier, ensuring that transformations are applied consistently.
    *   `LogisticRegression`: Chosen as the classification model, configured with `class_weight='balanced'` to handle the imbalanced nature of fraud detection datasets (where fraudulent transactions are rare).
    *   `train_test_split`: Divides the dataset into training and testing sets to evaluate model performance on unseen data.

### 5.Models Trained
Explores various machine learning models for fraud detection

1.  **Model 1: Logistic Regression (No balancing)**
    *   A baseline Logistic Regression model without any explicit handling of class imbalance.

2.  **Model 2: Logistic Regression with SMOTE**
    *   Logistic Regression combined with SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance by over-sampling the minority class.

3.  **Model 3: Random Forest (No balancing)**
    *   A baseline Random Forest Classifier without any explicit handling of class imbalance.

4.  **Model 4: Random Forest with SMOTE**
    *   Random Forest Classifier combined with SMOTE to address class imbalance.

### Evaluation Metrics
The models were evaluated using the following metrics, with a focus on detecting the minority class (fraudulent transactions):

*   **Precision (for class 1.0 - Fraud)**: The proportion of correctly identified fraudulent transactions among all transactions predicted as fraudulent.
*   **Recall (for class 1.0 - Fraud)**: The proportion of actual fraudulent transactions that were correctly identified.
*   **F1-Score (for class 1.0 - Fraud)**: The harmonic mean of precision and recall, providing a single metric that balances both.
*   **ROC AUC**: The Area Under the Receiver Operating Characteristic curve, which measures the model's ability to distinguish between classes across various thresholds.

### Model Results Comparison
The evaluation results were as follows:

| Model Name                                    | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | ROC AUC |
| :-------------------------------------------- | :---------------- | :------------- | :--------------- | :------ |
| Model 1: Logistic Regression (No balancing)   | 1.000             | 0.022          | 0.043            | 0.924   |
| Model 2: Logistic Regression with SMOTE       | 0.003             | 0.957          | 0.006            | 0.965   |
| Model 3: Random Forest (No balancing)         | 0.962             | 0.543          | 0.694            | 0.890   |
| Model 4: Random Forest with SMOTE             | 0.093             | 0.674          | 0.163            | 0.948   |

### Conclusion
Based on the evaluation metrics, **Model 3: Random Forest (No balancing)** performed well for fraud detection. It achieved a high precision (0.962) and a reasonable recall (0.543) for the fraudulent class, resulting in a good F1-score (0.694) and an overall good ROC AUC of 0.890. While Model 2 (Logistic Regression with SMOTE) had a higher ROC AUC, its extremely low precision (0.003) indicates a high rate of false positives, making it less practical for real-world fraud detection where minimizing false alarms is often crucial. Model 4 (Random Forest with SMOTE) had better recall but significantly lower precision and F1-score than Model 3.

Therefore, **Random Forest without balancing** was selected as the preferred model due to its balanced performance across precision, recall, and ROC AUC, demonstrating its ability to effectively identify fraud without an excessive number of false positives.

### 6. Model Persistence
*   **`joblib`**: The trained machine learning pipeline is saved to a file (`fraud_detection.pkl`) using `joblib.dump()`. This allows for the model to be loaded and reused later without needing to retrain it, facilitating deployment and future predictions.
### streamlitML.py

The streamlitML.py file is an interactive web application built with Streamlit that predicts financial fraud using a pre-trained machine learning model `(fraud_detection.pkl)`.
Its primary feature is a user-friendly interface that allows users to evaluate transaction safety without writing any code.












