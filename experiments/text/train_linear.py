import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

split = 0.8

df = pd.read_csv('./data/processed/Doc2Vec.csv')

print('Number of samples:', len(df))

X = df.loc[:, df.columns != 'Class'].values
y = df.loc[:, 'Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)

model = LogisticRegression()
model.fit(X_train, y_train)
accuracy_score(y_test, model.predict(X_test))