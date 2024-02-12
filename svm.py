import pandas as pd
import numpy as np
import optuna
import time

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score

def objective(trial):
    # activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh'])
    C = trial.suggest_int('C', 1, 150)
    gamma = trial.suggest_float('gamma', 0.001, 1)
    # p = trial.suggest_float('p', 1, 5)


    model = SVC(gamma=gamma, C=C)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    scoreMean = score.mean()

    return scoreMean


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

# num_rows_to_use = 5000
# X_subset = X[:num_rows_to_use]
# y_subset = y[:num_rows_to_use]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

# Create SVM model
# C57
svm_model = SVC(kernel = 'linear', C = 1.0)

start_time = time.time()
svm_model.fit(X_train, y_train)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
# trainaccuracy = accuracy_score(y_train, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1Score = f1_score(y_test, y_pred, average='weighted')
# confusionMatrix = confusion_matrix(y_test, predictions, labels=[0,1])
# precision = precision_score(y_test, predictions)

# print("Train Accuracy: ", trainaccuracy)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1Score)
# # Plot learning curve
# train_sizes, train_scores, test_scores = learning_curve(
#     svm_model, X_subset, y_subset, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
# )

# # Calculate mean and standard deviation for training and test sets
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)

# Plot the learning curve
train_sizes, train_scores, test_scores = learning_curve(
    svm_model, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve')
plt.legend()


# Kernel types to evaluate
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
scores = []

# # Loop through each kernel type and perform cross-validation
# for kernel in kernels:
#     model = SVC(kernel=kernel)
#     score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
#     scores.append(score)

# # Plotting the accuracies
# plt.figure(figsize=(10, 6))
# plt.bar(kernels, scores, color=['blue', 'green', 'red', 'purple'])
# plt.xlabel('Kernel Type')
# plt.ylabel('Accuracy')
# plt.title('SVM Model Accuracy by Kernel Type')
# plt.ylim(0, 1) # Assuming accuracy scores will be between 0 and 1
# plt.show()

# Validation Curve
param_range = np.array([0.01, 1, 5, 10, 20, 30, 40])
train_scores, test_scores = validation_curve(
    svm_model, X_train, y_train, param_name='gamma', param_range=param_range, cv=3, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('Gamma')
plt.ylabel('Accuracy Score')
plt.title('Validation Curve')
plt.legend()

param_range = np.array([1, 10, 30, 50, 70])
train_scores, test_scores = validation_curve(
    svm_model, X_train, y_train, param_name='C', param_range=param_range, cv=3, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('C')
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

fig2 = optuna.visualization.plot_slice(study, params=['n_neighbors', 'leaf_size', 'p'])
fig2.show()

fig3 = optuna.visualization.plot_param_importances(study)
fig3.show()