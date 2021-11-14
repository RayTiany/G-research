import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("train.csv")

newdf = df[(df["Asset_ID"] == 0)][:10000]
newdf.dropna(axis = 0, inplace = True)
newdf = newdf.reset_index(drop=True)

TIME_STEPS = 64
del newdf['Asset_ID']

train = newdf[:int(newdf.shape[0]*0.7)].copy().set_index('timestamp')
test = newdf[int(newdf.shape[0]*0.7):].copy().set_index('timestamp')

min_max = MinMaxScaler(feature_range=(0, 1))
train_scaled = min_max.fit_transform(train)

x_train = []
y_train = []
for i in range(TIME_STEPS, train_scaled.shape[0]):
    x_train.append(train_scaled[i - TIME_STEPS:i, :])
    y_train.append(train_scaled[i, :])
x_train, y_train = np.array(x_train), np.array(y_train)


total_data = pd.concat((train, test), axis=0)
inputs = total_data[len(total_data) - len(test) - TIME_STEPS:]
test_scaled = min_max.fit_transform(inputs)

x_test = []
y_test = []
for i in range(TIME_STEPS, test_scaled.shape[0]):
    x_test.append(test_scaled[i - TIME_STEPS:i, :])
    y_test.append(test_scaled[i, :])

x_test, y_test = np.array(x_test), np.array(y_test)


model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1:])))
model.add(Dropout(0.2))

model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128))
model.add(Dropout(0.2))

model.add(Dense(8))

model.compile(optimizer="adam", loss='mse',metrics=['mean_squared_error'])

history = model.fit(x_train, y_train,
                    batch_size = 256,
                    epochs=15)

plt.plot(history.history['loss'],label='training loss')