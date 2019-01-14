import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

window_size = 3
batch_size = 128
epochs_num = 32
split = 0.8

df = pd.read_csv('./data/processed/GloVe.csv')

print('Number of samples:', len(df))

X = df.loc[:, df.columns != 'Class'].values
y = to_categorical(df.loc[:, 'Class'])

X = np.concatenate((X[1:], y[:-1]), axis=1)
y = y[1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, shuffle=False)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='sigmoid'))

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size)
print(model.evaluate(X_test, y_test))

y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)

print('Confusion matrix')
print(confusion_matrix(y_true, y_pred))
