import optuna
import time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier

dataset = pd.read_csv('bank-additional-full.csv')
# dataset = dataset.sample(frac=1, random_state=0) # "frac=1" shuffles the entire DataFrame and random_state ensures we keep the same shuffle
dataset = dataset.drop('pdays', axis=1)


# X = dataset.iloc[:, :-1]
X = dataset.iloc[:, :15]
y = dataset.iloc[:, -1]

#Converting the data
Le = LabelEncoder()
for col in X.columns:
    # Check if the column data type is object (string)
    if X[col].dtype == 'object':
        # Use LabelEncoder to transform the string values to numerical values
        X[col] = Le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

baseclf = DecisionTreeClassifier(max_depth=5)

adaclf = AdaBoostClassifier(baseclf, n_estimators=50, learning_rate=0.1)

start_time = time.time()
fitadaclf = adaclf.fit(X_train, y_train)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

y_pred = fitadaclf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1Score = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1Score:.4f}")


# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    adaclf, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve')
plt.legend()


# Validation Curve
param_range = np.arange(1, 501, 50)
train_scores, test_scores = validation_curve(
    adaclf, X_train, y_train, param_name='n_estimators', param_range=param_range, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('N Estimators')
plt.ylabel('Accuracy Score')
plt.title('Validation Curve')
plt.legend()


param_range = np.array([0.0001, 0.001, 0.01, 0.1, 1])
train_scores, test_scores = validation_curve(
    adaclf, X_train, y_train, param_name='learning_rate', param_range=param_range, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy Score')
plt.title('Validation Curve')
plt.legend()

plt.show()
