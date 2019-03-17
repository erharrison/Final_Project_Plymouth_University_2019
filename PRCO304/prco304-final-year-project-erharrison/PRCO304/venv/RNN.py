import numpy
import pandas
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.python.ops import rnn, rnn_cell

look_back = 1

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

# normalize the dataset - batch normalisation???
scaler = MinMaxScaler(feature_range=(0, 1))  # MinMaxScalar is from ssikit learn library
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.15)  # 15%
test_size = int(len(dataset) * 0.15)  # 15%
train, test = dataset[0:train_size, :], dataset[train_size:(2*train_size), :]
print(len(train), len(test))

trainX, trainY = train[1:-1, :], train[2:, :]
testX, testY = test[1:-1, :], test[2:, :]

# (batch_size, timesteps, features)
trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
trainY = trainY.reshape(trainY.shape[0], 1, trainY.shape[1])
testX = testX.reshape(testX.shape[0], 1, testX.shape[1])
testY = testY.reshape(testY.shape[0], 1, testY.shape[1])

# create RNN model
model = Sequential()
model.add(
keras.layers.SimpleRNN(77, activation='relu', use_bias=True, kernel_initializer='he_normal', return_sequences=True))
model.add(Dense(77, activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy',
                       'mean_squared_error',  # linear regression performance measure
                       'mean_squared_logarithmic_error',  # used to measure difference between actual and predicted
                       'mean_absolute_error'])  # measure how close predictions are to output])

model.fit(trainX, trainY, epochs=1000, batch_size=len(trainX), verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print(trainPredict)
print(testPredict)