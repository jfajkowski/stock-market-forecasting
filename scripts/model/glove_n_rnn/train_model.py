import pandas as pd
from keras.layers import Dense, CuDNNGRU
from keras.models import Sequential
from keras.utils import to_categorical
from keras_preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split

batch_size = 128
epochs_num = 30
split = 0.8

df = pd.read_csv('./data/processed/GloVe.csv')

print('Number of samples:', len(df))

X = df.loc[:, df.columns != 'Class'].values
y = to_categorical(df.loc[:, 'Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, shuffle=False)

for window_size in [1, 3, 7]:
    train_generator = TimeseriesGenerator(X_train, y_train,
                                          length=window_size,
                                          batch_size=batch_size)
    test_generator = TimeseriesGenerator(X_test, y_test,
                                         length=window_size,
                                         batch_size=1)

    model = Sequential()
    model.add(CuDNNGRU(128, input_shape=(window_size, X_train.shape[1],)))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # Run training
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_generator, epochs=epochs_num)
    print(model.evaluate_generator(test_generator))
    model.save('./models/glove_{}_rnn.h5'.format(window_size))
