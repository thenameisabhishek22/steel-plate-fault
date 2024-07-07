# Faults Detection Using Machine Learning Models

This project aims to detect faults using several machine learning models such as Logistic Regression, Support Vector Classifier (SVC), and K-Nearest Neighbors (KNN). The dataset used for this project is obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA).

## Dataset

The dataset contains various features representing faults. It can be loaded directly using the following URL:

## Code Explanation

1. **Loading Data**:
    - The data is loaded using `pandas.read_csv` with tab-separated values.
    - Missing values are checked to ensure data integrity.

2. **Preprocessing**:
    - The target variable is encoded using `LabelEncoder`.
    - The dataset is split into training and testing sets using `train_test_split`.

3. **Model Training and Evaluation**:
    - Three models are trained: Logistic Regression, SVC, and KNN.
    - Each model's training and testing accuracy are computed.
    - Precision, recall, and F1-score are also evaluated.

4. **Model Comparison**:
    - Bar charts are plotted to compare the accuracy, precision, recall, and F1-score of the models.

5. **Hyperparameter Tuning**:
    - Grid search is used to tune the hyperparameters of the KNN model.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- numpy

Install the required libraries using the following command:
```bash
pip install pandas scikit-learn matplotlib numpy
