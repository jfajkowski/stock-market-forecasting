import numpy as np
import pandas as pd
from keras.layers import Dense, CuDNNGRU
from keras.models import Sequential
from keras.utils import to_categorical
from keras_preprocessing.sequence import TimeseriesGenerator
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

train_generator = TimeseriesGenerator(X_train, y_train,
                                      length=window_size,
                                      batch_size=batch_size)
test_generator = TimeseriesGenerator(X_test, y_test,
                                     length=window_size,
                                     batch_size=1)

model = Sequential()
model.add(CuDNNGRU(128, input_shape=(window_size, X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='sigmoid'))

# Run training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, epochs=epochs_num)
print(model.evaluate_generator(test_generator))

y_true = np.argmax(y_test[window_size:], axis=1)
y_pred = np.argmax(model.predict_generator(test_generator), axis=1)

print('Confusion matrix')
print(confusion_matrix(y_true, y_pred))
