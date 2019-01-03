import copy
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# %% Load corpus data
df = pd.read_csv('./data/interim/Classes_Changed.csv')
corpus = df.loc[:, 'Top1':'Top25']


# %% Remove newlines, bounding apostrophes/quotation marks and ensure that there are only single non-trailing spaces
def extract(document):
    if not isinstance(document, str):
        return ''

    document = document.replace('\n', ' ')
    document = re.sub(r" +", " ", document)
    match = re.match(r'^b\'(.+?)\'$|^b\"(.+?)\"$|(.+)', document)
    return next(g for g in match.groups() if g is not None) if match else ''


corpus = corpus.applymap(extract)


# %% Process all documents and abbreviate adequate words
class Trie():
    class Node():
        def __init__(self, value=None):
            self._children = dict()
            self._value = value

        @property
        def value(self):
            return self._value

        def insert(self, key, value, default=None):
            if len(key) > 0:
                if key[0] not in self._children:
                    self._children[key[0]] = Trie.Node(value=copy.deepcopy(default))
                self._children[key[0]].insert(key[1:], value, default)
            else:
                self._value = value

        def get(self, key, longest_match=False):
            if len(key) > 0:
                if key[0] not in self._children:
                    if longest_match:
                        return self._value
                    else:
                        raise KeyError()

                value = self._children[key[0]].get(key[1:], longest_match)
                return value if value else self._value
            else:
                return self._value

        def __contains__(self, key):
            if len(key) > 0:
                if key[0] not in self._children:
                    return False
                return self._children[key[0]].__contains__(key[1:])
            else:
                return True

        def is_leaf(self, key):
            if len(key) > 0:
                if key[0] not in self._children:
                    raise KeyError()
                return self._children[key[0]].is_leaf(key[1:])
            else:
                return not self._children

    def __init__(self, default=None, longest_match=False):
        self._default = default
        self._longest_match = longest_match
        self._root = Trie.Node(default)

    def __setitem__(self, key, value):
        self._root.insert(key, value, self._default)

    def __getitem__(self, key):
        return self._root.get(key, self._longest_match)

    def __contains__(self, key):
        return self._root.__contains__(key)

    def is_leaf(self, key):
        return self._root.is_leaf(key)


SPLIT_PATTERN = re.compile(r'(\W+)')
JOIN_PATTERN = ''
TRIE = Trie(longest_match=True)

with open('./data/external/abbreviations.csv') as f_in:
    for line in f_in:
        line = line.rstrip('\n')
        phrase, abbreviation = list(map(SPLIT_PATTERN.split, line.split(',')))
        TRIE[phrase] = (phrase, abbreviation)


def abbreviate(text):
    wrong_words = SPLIT_PATTERN.split(text)
    correct_words = []

    i = 0
    while i < len(wrong_words):
        result = TRIE[wrong_words[i:]]
        if result:
            current_wrong_words, current_correct_words = result
            i += len(current_wrong_words)
        else:
            current_correct_words = wrong_words[i]
            i += 1
        correct_words += current_correct_words

    return JOIN_PATTERN.join(correct_words)


corpus = corpus.applymap(abbreviate)

# %% Tokenize documents using NLTK
corpus = corpus.applymap(lambda document: word_tokenize(document))

# %% Lowercase documents
corpus = corpus.applymap(lambda document: [word.lower() for word in document])

# %% Remove stopwords
stopwords = stopwords.words('english')
corpus = corpus.applymap(lambda document: [word for word in document if word not in stopwords])

# %% Use lemmatizer to reduce dimensionality
lemmatizer = WordNetLemmatizer()
corpus = corpus.applymap(lambda document: [lemmatizer.lemmatize(word) for word in document])


# %% Print corpus statistics
def stats(corpus):
    document_lengths = [len(document) for document in corpus]
    tokens = [token for document in corpus for token in document]
    unique_tokens = set(tokens)
    print("Number of documents: {}".format(len(corpus)))
    print("Document length mean: {:.2f}".format(np.mean(document_lengths)))
    print("Document length variance: {:.2f}".format(np.var(document_lengths)))
    print("Number of tokens: {}".format(len(tokens)))
    print("Number of token types: {}".format(len(unique_tokens)))
    print("Type-Token Ratio: {:.2%}".format(len(unique_tokens) / len(tokens)))
    print()


stats(corpus.values.flatten())

# %% Persist cleaned data
df.loc[:, 'Top1':'Top25'] = corpus
df.to_csv(path_or_buf='./data/interim/Corpus_Cleaned.csv', index=False)
