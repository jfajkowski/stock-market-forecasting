import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import CuDNNGRU, Dense
from keras.utils import to_categorical
from keras_preprocessing.sequence import TimeseriesGenerator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

window_size = 3
batch_size = 16
epochs = 4

df = pd.read_csv('./data/interim/Corpus_Cleaned.csv')

raw = df.loc[:, 'Top1':'Top25'].apply(lambda x: ' '.join([str(s) for s in x]), axis=1)
y = to_categorical(df.loc[:, 'Class'])

raw_train, raw_test, y_train, y_test = train_test_split(raw, y, train_size=0.8, shuffle=False)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(raw_train)
X_test = vectorizer.transform(raw_test)

transformer = TfidfTransformer(use_idf=False)
X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)

X_train = X_train.todense()
X_test = X_test.todense()

train_generator = TimeseriesGenerator(X_train, y_train, length=window_size,
                                     batch_size=batch_size, shuffle=False)
test_generator = TimeseriesGenerator(X_test, y_test, length=window_size,
                                    batch_size=1, shuffle=False)

model = Sequential()
model.add(CuDNNGRU(128, input_shape=(window_size, X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Run training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, epochs=epochs)
print(model.evaluate_generator(test_generator))

y_true = np.argmax(y_test[window_size:], axis=1)
y_pred = np.argmax(model.predict_generator(test_generator), axis=1)

print('Confusion matrix')
print(confusion_matrix(y_true, y_pred))
