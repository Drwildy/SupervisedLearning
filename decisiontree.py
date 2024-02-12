import optuna
import time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from sklearn.preprocessing import LabelEncoder

def objective(trial):
    min_samples_split = trial.suggest_int('min_samples_split', 100, 600)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 50)
    max_depth = trial.suggest_int('max_depth', 3, 6)

    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

# clf = DecisionTreeClassifier()
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=112, min_samples_leaf=6)
# clf = DecisionTreeClassifier(max_depth=5, min_samples_split=400, criterion='gini', min_samples_leaf=10)
# clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=100)
# fitclf = clf.fit(X_train, y_train)
start_time = time.time()
fitclf = clf.fit(X_train, y_train)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

fitclf.get_params()

predictions = fitclf.predict(X_test)

probapredictions = fitclf.predict_proba(X_test)[:, 1]

zerosaccuracy = accuracy_score(y_test, np.full(y_test.size, 'no'))
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1Score = f1_score(y_test, predictions, average='weighted')
auc = roc_auc_score(y_test, probapredictions)
zerosauc = roc_auc_score(y_test, np.zeros_like(y_test))
# confusionMatrix = confusion_matrix(y_test, predictions, labels=[0,1])
# precision = precision_score(y_test, predictions)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1Score)
print("AUC: ",auc)
print('Zeros AUC: ', zerosauc)

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    fitclf, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.05, 1.0, 20))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve')
plt.legend()

# Validation Curve
param_range = np.arange(1, 100)
train_scores, test_scores = validation_curve(
    clf, X_train, y_train, param_name='min_samples_leaf', param_range=param_range, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('Min Samples Leaf')
plt.ylabel('Accuracy Score')
plt.title('Validation Curve')
plt.legend()


param_range = np.arange(1, 10)
train_scores, test_scores = validation_curve(
    clf, X_train, y_train, param_name='max_depth', param_range=param_range, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy Score')
plt.title('Validation Curve')
plt.legend()

# # Validation Curve for min_samples_split
param_range = np.arange(2, 12)
param_range = np.array([50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
train_scores, test_scores = validation_curve(
    clf, X_train, y_train, param_name='min_samples_split', param_range=param_range, cv=5, scoring='accuracy')


plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.xlabel('Min Samples Split')
plt.ylabel('Accuracy Score')
plt.title('Validation Curve for Min Samples Split')
plt.legend()
plt.show()

# ----- Optuna-------------# #

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(42))

study.optimize(objective, n_trials=200)

print(study.best_params)
best_params = study.best_params

fig = optuna.visualization.plot_optimization_history(study)
fig.show()

fig2 = optuna.visualization.plot_slice(study, params=['criterion', 'min_samples_leaf', 'min_samples_split'])
fig2.show()

fig3 = optuna.visualization.plot_param_importances(study)
fig3.show()