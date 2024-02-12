import optuna
import time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from sklearn.preprocessing import LabelEncoder


def objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 25)
    # leaf_size = trial.suggest_int('leaf_size', 10, 1000)
    p = trial.suggest_int('p', 1, 5)


    model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    scoreMean = score.mean()

    return scoreMean

dataset = pd.read_csv('insurance.csv')

bins = [0, 7500, 15000, float('inf')]
labels = ['low', 'medium', 'high']

dataset['chargesClass'] = pd.cut(dataset['charges'], bins=bins, labels=labels, right=False)
dataset = dataset.drop(['charges'], axis=1)
# print(dataset)

X = dataset.drop(['chargesClass'], axis=1)
y = dataset['chargesClass']

# print(y.value_counts())

Le = LabelEncoder()
for col in X.columns:
    # Check if the column data type is object (string)
    if X[col].dtype == 'object':
        # Use LabelEncoder to transform the string values to numerical values
        X[col] = Le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

knn = KNeighborsClassifier(p=1, n_neighbors=8)

start_time = time.time()
knn = knn.fit(X_train, y_train)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

y_pred = knn.predict(X_test)

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
    knn, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve')
plt.legend()


# Validation Curve
param_range = np.arange(1, 20, 1)
train_scores, test_scores = validation_curve(
    knn, X_train, y_train, param_name='n_neighbors', param_range=param_range, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('N Neighbors')
plt.ylabel('Accuracy Score')
plt.title('Validation Curve')
plt.legend()

param_range = np.arange(50, 1000, 50)
train_scores, test_scores = validation_curve(
    knn, X_train, y_train, param_name='leaf_size', param_range=param_range, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('Leaf Size')
plt.ylabel('Accuracy Score')
plt.title('Validation Curve')
plt.legend()

param_range = np.arange(1, 10, 1)
train_scores, test_scores = validation_curve(
    knn, X_train, y_train, param_name='p', param_range=param_range, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('P')
plt.ylabel('Accuracy Score')
plt.title('Validation Curve')
plt.legend()


plt.show()

# ----- Optuna-------------# #

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(42))

study.optimize(objective, n_trials=100)

print(study.best_params)
best_params = study.best_params

fig = optuna.visualization.plot_optimization_history(study)
fig.show()

fig2 = optuna.visualization.plot_slice(study, params=['n_neighbors', 'p'])
fig2.show()

fig3 = optuna.visualization.plot_param_importances(study)
fig3.show()