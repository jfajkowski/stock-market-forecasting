import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

batch_size = 16
epochs = 4
split = 0.8

df = pd.read_csv('./data/interim/Classes_Changed.csv')

raw = df.loc[:, 'Top1':'Top25'].apply(lambda x: ' '.join([str(s) for s in x]), axis=1)
y = to_categorical(df.loc[:, 'Class'])

tokens = set(' '.join(raw.values.flatten()).split(' '))
tokens = dict(zip(tokens, list(range(len(tokens)))))

X = raw.apply(lambda document: [tokens[word] for word in document.split(' ')])

max_len = max(X.apply(len))

X = pad_sequences(X, maxlen=max_len, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, shuffle=False)


model = Sequential()
model.add(Embedding(input_dim=len(tokens), output_dim=256, input_length=max_len))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
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