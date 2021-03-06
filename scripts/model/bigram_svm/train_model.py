import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# %% Load data
df = pd.read_csv('./data/interim/Classes_Changed.csv')

raw = df.loc[:, 'Top1':'Top25'].apply(lambda x: ' '.join([str(s) for s in x]), axis=1)
y = df['Class']

# %% Split it into test and training sets
raw_train, raw_test, y_train, y_test = train_test_split(raw, y, train_size=0.8, shuffle=False)

# %% Define model
clf = SVC(decision_function_shape='ovo', kernel='rbf', probability=True)
model = Pipeline([
    ('vect', CountVectorizer(ngram_range=(2, 2), stop_words='english')),
    ('tfidf', TfidfTransformer(use_idf=False)),
    ('scale', StandardScaler(with_mean=False)),
    ('clf', clf),
])
model.fit(raw_train, y_train)
accuracy_score(y_test, model.predict(raw_test))

# %% Persist it
import pickle
with open('./models/raw_bigram_svm.mdl', mode='wb') as model_file:
    pickle.dump(model, model_file)



