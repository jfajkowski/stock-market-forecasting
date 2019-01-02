import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/processed/Stripped.csv')

raw = df.loc[:, 'Top1':'Top25'].apply(lambda x: ' '.join([str(s) for s in x]), axis=1)
y = df['Label']

raw_train, raw_test, y_train, y_test = train_test_split(raw, y, train_size=0.8, shuffle=False)

vectorizer = CountVectorizer(ngram_range=(2, 2))
X_train = vectorizer.fit_transform(raw_train)
X_test = vectorizer.transform(raw_test)

model = LogisticRegression()
model.fit(X_train, y_train)
accuracy_score(y_test, model.predict(X_test))