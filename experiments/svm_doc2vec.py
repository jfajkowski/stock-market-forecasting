import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

df = pd.read_csv('./data/processed/Doc2Vec.csv', lineterminator='\n', sep=',')
df.columns = df.columns.str.strip()

raw = df.iloc[:, 0:128]
y = df['Class']

raw_train, raw_test, y_train, y_test = train_test_split(raw, y, train_size=0.8, shuffle=False)

clf = SVC(decision_function_shape='ovr', kernel='linear')

model = Pipeline([
    ('clf', clf),
])
model.fit(raw_train, y_train)
print(accuracy_score(y_test, model.predict(raw_test)))