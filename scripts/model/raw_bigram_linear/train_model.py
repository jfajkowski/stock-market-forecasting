import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('./data/interim/Classes_Changed.csv')

raw = df.loc[:, 'Top1':'Top25'].apply(lambda x: ' '.join([str(s) for s in x]), axis=1)
y = df['Class']

# Split it into training and test set
raw_train, raw_test, y_train, y_test = train_test_split(raw, y, train_size=0.8, shuffle=False)

# Prepare model
vectorizer = CountVectorizer(ngram_range=(2, 2))
X_train = vectorizer.fit_transform(raw_train)
X_test = vectorizer.transform(raw_test)

classifier = LogisticRegression(solver='lbfgs', multi_class='multinomial')
classifier.fit(X_train, y_train)
accuracy_score(y_test, classifier.predict(X_test))

# Persist it
import pickle
with open('./models/raw_bigram_linear.vcr', mode='wb') as vcr_file:
    pickle.dump(vectorizer, vcr_file)
with open('./models/raw_bigram_linear.clf', mode='wb') as clf_file:
    pickle.dump(classifier, clf_file)