import pandas as pd
from keras import Sequential
from keras.layers import Dense, CuDNNGRU
from keras_preprocessing.sequence import TimeseriesGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

window_size = 7
batch_size = 256
epochs = 64

df = pd.read_csv('./data/raw/DJIA_table.csv')

scaler = StandardScaler()
data = scaler.fit_transform((df['Close'] - df['Open']).values.reshape(-1, 1))

X = data[:-1]
y = data[1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

train_data_gen = TimeseriesGenerator(X_train, y_train, length=window_size, batch_size=batch_size, shuffle=False)
test_data_gen = TimeseriesGenerator(X_test, y_test, length=window_size, batch_size=batch_size, shuffle=False)

model = Sequential()
model.add(CuDNNGRU(4, input_shape=(window_size, 1,)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit_generator(train_data_gen, epochs=epochs).history

index = [df['Open'][0]]
for i, d in enumerate(scaler.inverse_transform(data)):
    index.append(index[i] + d)

index_train = [df['Open'][0]]
for i, d in enumerate(scaler.inverse_transform(model.predict_generator(train_data_gen))):
    index_train.append(index_train[i] + d)

index_test = [index_train[-1]]
for i, d in enumerate(scaler.inverse_transform(model.predict_generator(test_data_gen))):
    index_test.append(index_test[i] + d)

begin = window_size
join = begin + len(index_train)
end = join + len(index_test)
plt.plot(index)
plt.plot(list(range(begin, join)), index_train)
plt.plot(list(range(join, end)), index_test)
plt.show()