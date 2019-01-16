import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/processed/GloVe.csv', lineterminator='\n', sep=',')
df.columns = df.columns.str.strip()

X = df.loc[:, df.columns != 'Class'].values
y = df['Class']

raw_train, raw_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)

model = LogisticRegression()
model.fit(raw_train, y_train)
y_pred = model.predict(raw_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))