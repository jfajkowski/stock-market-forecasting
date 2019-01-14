import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import CuDNNGRU, Dense
from keras.utils import to_categorical
from keras_preprocessing.sequence import TimeseriesGenerator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

batch_size = 8
epochs = 4

df = pd.read_csv('./data/interim/Corpus_Cleaned.csv')

raw = df.loc[:, 'Top1':'Top25'].apply(lambda x: ' '.join([str(s) for s in x]), axis=1)
y = to_categorical(df.loc[:, 'Class'])

raw_train, raw_test, y_train, y_test = train_test_split(raw, y, train_size=0.8, shuffle=False)

vectorizer = CountVectorizer(binary=True)
X_train = vectorizer.fit_transform(raw_train)
X_test = vectorizer.transform(raw_test)

transformer = TfidfTransformer()
X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)

X_train = X_train.todense()
X_test = X_test.todense()
y_train = np.matrix(y_train)
y_test = np.matrix(y_test)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='sigmoid'))

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
print(model.evaluate(X_test, y_test))

y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)

print('Confusion matrix')
print(confusion_matrix(y_true, y_pred))
