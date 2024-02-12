import optuna
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

def objective(trial):
    activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh'])
    hidden_layer_sizes = trial.suggest_int('hidden_layer_sizes', 100, 900)
    batch_size = trial.suggest_int('batch_size', 200, 1600)
    learning_rate_init = trial.suggest_float('learning_rate_init', 0.0001, 0.01)


    model = MLPClassifier(activation=activation, hidden_layer_sizes=hidden_layer_sizes, batch_size=batch_size, learning_rate_init=learning_rate_init)

    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    scoreMean = score.mean()

    return scoreMean


# Load the dataset
dataset = pd.read_csv('bank-additional-full.csv')
dataset = dataset.drop('pdays', axis=1)

# Select features and target variable
X = dataset.iloc[:, :15]
y = dataset.iloc[:, -1]

# Converting categorical data to numerical
Le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = Le.fit_transform(X[col])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

# Create an MLPClassifier
model = MLPClassifier((100,75,50,25), learning_rate_init=0.01, max_iter=5000, random_state=42)
# model = MLPClassifier(batch_size=200, hidden_layer_sizes=250, max_iter=5000, random_state=42)
# model = MLPClassifier(activation='tanh', hidden_layer_sizes=272, batch_size=1072, learning_rate_init=0.00094)


# Train the model
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
# trainaccuracy = accuracy_score(y_train, y_predtrain)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1Score = f1_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
# print("Train Accuracy: ", trainaccuracy)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1Score:.4f}")


# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve')
plt.legend()


# Validation Curve
param_range = [(100,), (50,50), (100,50,25), (100,75,50,25)]
train_scores, test_scores = validation_curve(
    model, X_train, y_train, param_name='hidden_layer_sizes', param_range=param_range, cv=5, scoring='accuracy')

# Convert tuple values to string for plotting purposes
param_range_str = [str(x) for x in param_range]

plt.figure(figsize=(10, 6))
plt.plot(param_range_str, np.mean(train_scores, axis=1), marker='o', label='Training Score')
plt.plot(param_range_str, np.mean(test_scores, axis=1), marker='s', label='Cross-Validation Score')
plt.xticks(param_range_str, param_range_str) # Set x-tick labels to clearly show different configurations
plt.xlabel('Hidden Layer Sizes')
plt.ylabel('Accuracy Score')
plt.title('Validation Curve for MLPClassifier')
plt.legend()

param_range = np.arange(100, 2000, 100)
train_scores, test_scores = validation_curve(
    model, X_train, y_train, param_name='batch_size', param_range=param_range, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('Batch Size')
plt.ylabel('Accuracy Score')
plt.title('Validation Curve')
plt.legend()

param_range = np.array([0.0001, 0.001, 0.01, 0.1])
train_scores, test_scores = validation_curve(
    model, X_train, y_train, param_name='learning_rate_init', param_range=param_range, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('Learning Rate')
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

fig2 = optuna.visualization.plot_slice(study, params=['hidden_layer_sizes', 'activation', 'min_samples_split'])
fig2.show()

fig3 = optuna.visualization.plot_param_importances(study)
fig3.show()