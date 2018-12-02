import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def extract(s):
    groups = re.match(r'^b\'(.+?)\'$|^b\"(.+?)\"$|(.+)', s.lower().replace('\n', '')).groups()
    return next(g for g in groups if g is not None)

df = pd.read_csv('./data/raw/RedditNews.csv')

for i, row in df.iterrows():
    row['Processed'] = extract(row['News'])

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df['News'])

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

for i, row in df.iterrows():
    row['Vector'] = X_train_tf[i]