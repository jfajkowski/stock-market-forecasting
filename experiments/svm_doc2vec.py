import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/data/processed/Doc2Vec.csv', lineterminator='\n', sep=',')
df.columns = df.columns.str.strip()

print('Number of samples:', len(df))

X = df.loc[:, df.columns != 'Class'].values
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)

clf = SVC(decision_function_shape='ovr', kernel='linear')

model = Pipeline([
    ('scale', StandardScaler(with_mean=False)),
    ('clf', clf),
])
model.fit(X_train, y_train)
print(accuracy_score(y_test, model.predict(X_test)))