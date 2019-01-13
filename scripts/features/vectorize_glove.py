import numpy as np
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_lg")

df = pd.read_csv('./data/interim/Corpus_Cleaned.csv')

df['Vector'] = df.loc[:, 'Top1':'Top25'].apply(lambda x: nlp(' '.join([str(s) for s in x])).vector, axis=1)

matrix = np.zeros((len(df), df['Vector'][0].shape[0]))
for i, r in df.iterrows():
    matrix[i, ] = r['Vector']

df_out = pd.DataFrame(data=matrix)
df_out['Class'] = df['Class']

df_out.to_csv(path_or_buf='./data/processed/GloVe.csv', index=False)