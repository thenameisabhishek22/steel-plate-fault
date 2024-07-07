# -*- coding: utf-8 -*-
"""SteelPlateFaults

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1r1Q8tY_yPPr6wAkopXrX7omDqJZ3eRwY

Data Preprocessing
"""

import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA"
data = pd.read_csv(url, sep='\t', header=None)

print(data.shape)

data.head()

print(data.isnull().sum())

data.head()

"""Label Encoding"""

input = data.drop(columns=[27])
target = data[27]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
target = le.fit_transform(target)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input, target, test_size=0.3, random_state=42)
print(x_train.shape)
print(x_test.shape)

print(target)

"""Logistic Regression"""

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)

ytrain_lr=lr.predict(x_train)

ytest_lr=lr.predict(x_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Training Accuracy of Logistic Regression Model:", accuracy_score(y_train, ytrain_lr))

print("Testing Accuracy of Logistic Regression Model:", accuracy_score(y_test, ytest_lr))

precision = precision_score(y_test, ytest_lr, zero_division=1)
recall = recall_score(y_test, ytest_lr)
f1 = f1_score(y_test, ytest_lr)

print("Precision of Logistic Regression Model:", precision)
print("Recall of Logistic Regression Model:", recall)
print("F1-Score of Logistic Regression Model:", f1)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train, ytrain_lr))

print(confusion_matrix(y_test, ytest_lr))

"""Support Vector Classification (SVC)"""

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)

ytrain_svc=svc.predict(x_train)

ytest_svc=svc.predict(x_test)

print("Training Accuracy of SVC Model:", accuracy_score(y_train, ytrain_svc))

print("Testing Accuracy of SVC Model:", accuracy_score(y_test, ytest_svc))

precision = precision_score(y_test, ytest_svc)
recall = recall_score(y_test, ytest_svc)
f1 = f1_score(y_test, ytest_svc)

print("Precision of SVC Model:", precision)
print("Recall of SVC Model:", recall)
print("F1-Score of SVC Model:", f1)

print(confusion_matrix(y_train, ytrain_svc))

print(confusion_matrix(y_test, ytest_svc))

"""K-Nearest Neighbors"""

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train, y_train)

ytrain_knn=knn.predict(x_train)

ytest_knn=knn.predict(x_test)

print("Training Accuracy of DecisionTreeClassifier Model:", accuracy_score(y_train, ytrain_knn))

print("Testing Accuracy of DecisionTreeClassifier Model:", accuracy_score(y_test, ytest_knn))

precision = precision_score(y_test, ytest_knn)
recall = recall_score(y_test, ytest_knn)
f1 = f1_score(y_test, ytest_knn)

print("Precision of KNN Model:", precision)
print("Recall of KNN Model:", recall)
print("F1-Score of KNN Model:", f1)

print(confusion_matrix(y_train, ytrain_knn))

print(confusion_matrix(y_test, ytest_knn))

"""HyperParameter Tuning on KNN Model"""

from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
y_pred = grid_search.best_estimator_.predict(x_test)
print(accuracy_score(y_test, y_pred))

"""Compare using graph plots"""

import matplotlib.pyplot as plt
import numpy as np

model_names = ['Logistic Regression', 'SVC', 'KNN']

training_accuracies = [accuracy_score(y_train, ytrain_lr), accuracy_score(y_train, ytrain_svc), accuracy_score(y_train, ytrain_knn)]

testing_accuracies = [accuracy_score(y_test, ytest_lr), accuracy_score(y_test, ytest_svc), accuracy_score(y_test, ytest_knn)]

x = np.arange(len(model_names))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, training_accuracies, width, label='Training Accuracy')
rects2 = ax.bar(x + width/2, testing_accuracies, width, label='Testing Accuracy')

ax.set_ylabel('Accuracy')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()

plt.show()

model_names = ['Logistic Regression', 'SVC', 'KNN']

precision_scores = [precision_score(y_test, ytest_lr), precision_score(y_test, ytest_svc), precision_score(y_test, ytest_knn)]

recall_scores = [recall_score(y_test, ytest_lr), recall_score(y_test, ytest_svc), recall_score(y_test, ytest_knn)]

f1_scores = [f1_score(y_test, ytest_lr), f1_score(y_test, ytest_svc), f1_score(y_test, ytest_knn)]

x = np.arange(len(model_names))
width = 0.25
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, precision_scores, width, label='Precision')
rects2 = ax.bar(x, recall_scores, width, label='Recall')
rects3 = ax.bar(x + width, f1_scores, width, label='F1 Score')

ax.set_ylabel('Score')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()

plt.show()
