import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, LSTMCell, Activation
from sklearn.preprocessing import MinMaxScaler
import numpy
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import time

from Cell import MinimalRNNCell

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv(
    r"C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\Data.csv",
    delimiter=",",
    engine='python')

dataset = dataframe.values
# floating point values are more suitable for neural networks
dataset = dataset.astype('float32')


scaler = MinMaxScaler(feature_range=(0, 1))  # MinMaxScalar is from scikit learn library
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.8)  # 80%
test_size = int(len(dataset) * 0.2)  # 20%
train, test = dataset[0:train_size, :], dataset[train_size:(2*train_size), :]
print(len(train), len(test))

trainX, trainY = train[1:-1, :], train[2:, :]
testX, testY = test[1:-1, :], test[2:, :]

# (batch_size, timesteps, features)
trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
trainY = trainY.reshape(trainY.shape[0], 1, trainY.shape[1])
testX = testX.reshape(testX.shape[0], 1, testX.shape[1])
testY = testY.reshape(testY.shape[0], 1, testY.shape[1])

cell = MinimalRNNCell(32)
x = keras.Input((None, 5))

# create RNN model
# Sequential model is a linear stack of layers
model = Sequential()
model.add(keras.layers.RNN(cell, return_sequences=True))
model.add(Activation('relu'))
# model.add(keras.layers.SimpleRNN(77, activation='relu', use_bias=True, kernel_initializer='he_normal', return_sequences=True))
# model.add(LSTM(77, activation='relu', use_bias=True, kernel_initializer='he_normal', return_sequences=True))
model.add(Dense(77, activation='relu'))


model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_squared_error',  # linear regression performance measure
                       'mean_squared_logarithmic_error',  # used to measure difference between actual and predicted
                       'mean_absolute_error'])  # measure how close predictions are to output])

name = "recurrent-neural-network-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='TensorBoardResults/logs/{}'.format(name), histogram_freq=0,
                          write_graph=True)

model.fit(trainX, trainY, epochs=300, batch_size=len(trainX), verbose=1, callbacks=[tensorboard])
model.fit(testX, testY, epochs=300, batch_size=len(testX), verbose=1, callbacks=[tensorboard])

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print(trainPredict.shape)
print(testPredict.shape)

plt.imshow(trainPredict, interpolation='none')
plt.colorbar()
plt.show()