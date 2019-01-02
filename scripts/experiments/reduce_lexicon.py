import numpy as np
import pandas as pd
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer, LancasterStemmer


def stats(corpus):
    document_lengths = corpus.apply(len)
    tokens = [token for document in corpus for token in document]
    unique_tokens = set(tokens)
    print("Number of documents: {}".format(len(corpus)))
    print("Document length mean: {:.2f}".format(np.mean(document_lengths)))
    print("Document length variance: {:.2f}".format(np.var(document_lengths)))
    print("Number of tokens: {}".format(len(tokens)))
    print("Number of token types: {}".format(len(unique_tokens)))
    print("Type-Token Ratio: {:.2%}".format(len(unique_tokens) / len(tokens)))
    print()


# Load news data
df = pd.read_csv('./data/raw/RedditNews.csv')
corpus = df["News"]


# Remove newlines and ensure that there are only single non-trailing spaces
# Remove bounding apostrophes/quotation marks
def extract(document):
    document = document.replace('\n', ' ')
    document = re.sub(r" +", " ", document)
    match = re.match(r'^b\'(.+?)\'$|^b\"(.+?)\"$|(.+)', document)
    return next(g for g in match.groups() if g is not None) if match else ''


corpus = corpus.apply(extract)

# Documents split by space
stats(corpus.apply(lambda document: document.split(" ")))

# Documents tokenized with NLTK
corpus = corpus.apply(lambda document: word_tokenize(document))
stats(corpus)

# Lowercase documents
corpus = corpus.apply(lambda document: [word.lower() for word in document])
stats(corpus)

# Remove stopwords
def remove_stopwords(document):
    return [word for word in document if word not in stopwords.words('english')]

corpus = corpus.apply(remove_stopwords)
stats(corpus)

# Use stemmers and lemmatizer to reduce dimensionality
stemmers = [PorterStemmer(), LancasterStemmer(), SnowballStemmer('english')]
lemmatizer = WordNetLemmatizer()
corpora = [
    corpus.apply(lambda document: [stemmers[0].stem(word) for word in document]),
    corpus.apply(lambda document: [stemmers[1].stem(word) for word in document]),
    corpus.apply(lambda document: [stemmers[2].stem(word) for word in document]),
    corpus.apply(lambda document: [lemmatizer.lemmatize(word) for word in document])
]
for corpus in corpora:
    stats(corpus)