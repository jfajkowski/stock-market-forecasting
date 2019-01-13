import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

df = pd.read_csv('./data/interim/Corpus_Cleaned.csv')
headlines = df.loc[:, 'Top1':'Top25'].apply(lambda x: ' '.join([str(s) for s in x]), axis=1)
documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(headlines)]
model = Doc2Vec(documents, vector_size=128, window=3)

df['Vector'] = [model.docvecs[i] for i in range(len(headlines))]

matrix = np.zeros((len(df), df['Vector'][0].shape[0]))
for i, r in df.iterrows():
    matrix[i, ] = r['Vector']

df_out = pd.DataFrame(data=matrix)
df_out['Class'] = df['Class']

df_out.to_csv(path_or_buf='./data/processed/Doc2Vec.csv', index=False)