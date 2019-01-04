import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import CuDNNGRU, Dense
from keras_preprocessing.sequence import TimeseriesGenerator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

window_size = 2
batch_size = 8
epochs = 5

df = pd.read_csv('./data/processed/Stripped.csv')

raw = df.loc[:, 'Top1':'Top25'].apply(lambda x: ' '.join([str(s) for s in x]), axis=1)
y = df['Label']

raw_train, raw_test, y_train, y_test = train_test_split(raw, y, train_size=0.8, shuffle=False)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(raw_train)
X_test = vectorizer.transform(raw_test)

transformer = TfidfTransformer(use_idf=False)
X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)

X_train = X_train.todense()
X_test = X_test.todense()
y_train = np.matrix(y_train).transpose()
y_test = np.matrix(y_test).transpose()

train_generator = TimeseriesGenerator(X_train, y_train, length=window_size,
                                     batch_size=batch_size, shuffle=False)
test_generator = TimeseriesGenerator(X_test, y_test, length=window_size,
                                    batch_size=1, shuffle=False)

model = Sequential()
model.add(CuDNNGRU(4, input_shape=(window_size, X_train.shape[1],)))
model.add(Dense(1, activation='softmax'))

# Run training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, epochs=epochs)
model.evaluate_generator(test_generator)