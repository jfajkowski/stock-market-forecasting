import re
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords

def extract(s):
    groups = re.match(r'^b\'(.+?)\'$|^b\"(.+?)\"$|(.+)', s.lower().replace('\n', '')).groups()
    return next(g for g in groups if g is not None)

df = pd.read_csv('./data/raw/RedditNews.csv')
df['News'] = df['News'].apply(extract)

s = ' '.join(df['News'])
tokens = word_tokenize(s)
tokens_set = set(tokens)
tokens_counter = Counter(tokens)
tokens_without_stopwords_set = tokens_set.difference(stopwords.words('english'))
tokens_without_stopwords_counter = Counter({k:tokens_counter[k] for k in tokens_without_stopwords_set if k in tokens_counter})
tokens_filtered_set = set(filter(lambda t: len(t) > 2, tokens_without_stopwords_set))
tokens_filtered_counter = Counter({k:tokens_counter[k] for k in tokens_filtered_set if k in tokens_counter})
