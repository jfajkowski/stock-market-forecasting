import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

df = pd.read_csv('./data/processed/Stripped.csv')
headlines = df.loc[:, 'Top1':'Top25'].apply(lambda x: ' '.join([str(s) for s in x]), axis=1)
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(headlines)]
model = Doc2Vec(documents, vector_size=300, window=3)

df['Vector'] = list(map(lambda x: model.infer_vector(x.split()), headlines))

matrix = np.zeros((len(df), df['Vector'][0].shape[0]))
for i, r in df.iterrows():
    matrix[i, ] = r['Vector']

df_out = pd.DataFrame(data=matrix)
df_out['Class'] = df['Label']

df_out.to_csv(path_or_buf='./data/processed/Doc2Vec.csv', index=False)