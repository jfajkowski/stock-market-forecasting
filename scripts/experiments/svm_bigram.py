import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv('./data/processed/Stripped.csv')

raw = df.loc[:, 'Top1':'Top25'].apply(lambda x: ' '.join([str(s) for s in x]), axis=1)
y = df['Label']

raw_train, raw_test, y_train, y_test = train_test_split(raw, y, train_size=0.8, shuffle=False)

model = Pipeline([
    ('vect', CountVectorizer(ngram_range=(2, 2))),
    ('tfidf', TfidfTransformer(use_idf=False)),
    ('clf', SGDClassifier()),
])

model.fit(raw_train, y_train)
accuracy_score(y_test, model.predict(raw_test))