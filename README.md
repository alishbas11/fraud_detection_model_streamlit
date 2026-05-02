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

### 5. Model Evaluation
*   **`scikit-learn.metrics`**: Provides tools to assess the model's performance:
    *   `Pipe.score()`: Reports the overall accuracy of the model on the test set.
    *   `confusion_matrix()`: Generates a matrix showing true positives, true negatives, false positives, and false negatives, crucial for understanding classification errors.
    *   `classification_report()`: Offers a detailed summary of precision, recall, and f1-score for each class, which are more informative than accuracy for imbalanced datasets.

### 6. Model Persistence
*   **`joblib`**: The trained machine learning pipeline is saved to a file (`fraud_detection.pkl`) using `joblib.dump()`. This allows for the model to be loaded and reused later without needing to retrain it, facilitating deployment and future predictions.
### streamlitML.py

The streamlitML.py file is an interactive web application built with Streamlit that predicts financial fraud using a pre-trained machine learning model `(fraud_detection.pkl)`.
Its primary feature is a user-friendly interface that allows users to evaluate transaction safety without writing any code.

The application prompts users to select a transaction type `(e.g., CASH_OUT, TRANSFER)` and enter numerical values for the transaction amount, alongside the old and new balances of both the sender and receiver.

Upon clicking the `"Predict"` button, the app’s underlying logic structures these six inputs into a Pandas DataFrame. This formatted data is then passed to the loaded machine learning model for analysis. The model evaluates the transaction patterns and returns a binary prediction. Finally, the app displays the result visually: a red error message indicating "Fraud" if the output is 1, or a green success message confirming the transaction is safe if the output is 0.
