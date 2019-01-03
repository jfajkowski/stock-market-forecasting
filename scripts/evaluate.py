import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

results = pd.read_csv("./data/processed/Results.csv", sep=',')

y_true = results['Class']
y_pred = results['Prediction']

print('Accuracy')
print(accuracy_score(y_true, y_pred))

print('Evaluation report')
print(classification_report(y_true, y_pred))

print('Confusion matrix')
print(confusion_matrix(y_true, y_pred))
