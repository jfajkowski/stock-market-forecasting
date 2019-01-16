import numpy as np
import pandas as pd
import spacy

# %% Load spaCy model
nlp = spacy.load("en_core_web_lg")

# %% Load cleaned data
df = pd.read_csv('./data/interim/Corpus_Cleaned.csv')

# %% Extract document vectors
df['Vector'] = df.loc[:, 'Top1':'Top25'].apply(lambda x: nlp(' '.join([str(s) for s in x])).vector, axis=1)
matrix = np.zeros((len(df), df['Vector'][0].shape[0]))
for i, r in df.iterrows():
    matrix[i, ] = r['Vector']

# %% Prepare file with so many columns as in Doc2Vec vectors plus one more (for a label)
df_out = pd.DataFrame(data=matrix)
df_out['Class'] = df['Class']
df_out.to_csv(path_or_buf='./data/processed/GloVe.csv', index=False)