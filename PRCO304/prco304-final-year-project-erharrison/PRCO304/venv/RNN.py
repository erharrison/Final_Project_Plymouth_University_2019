import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, LSTMCell, Activation, RNN
from sklearn.preprocessing import MinMaxScaler
import numpy
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import time
from keras.utils.vis_utils import plot_model

from Cell import MinimalRNNCell


# fix random seed for reproducibility
seed = 0
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

trainX, trainY = train[1:-1, :], train[2:, :] # shape is 82
testX, testY = test[1:-1, :], test[2:, :] # shape is 20 - could be causing error when fitting?

# (batch_size, timesteps, features)
trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])  # trainX use fisrt row and see how that goes then use 20 rows etc.
trainY = trainY.reshape(trainY.shape[0], 1, trainY.shape[1])
testX = testX.reshape(testX.shape[0], 1, testX.shape[1])
testY = testY.reshape(testY.shape[0], 1, testY.shape[1])
print(trainX.shape, testX.shape)


cell = MinimalRNNCell(77)


# create recurrent neural network
model = Sequential()  # Sequential model is a linear stack of layers
#  model.add(Activation('relu'))
#  model.add(RNN(cell, return_sequences=True))
model.add(keras.layers.SimpleRNN(77, activation='tanh', use_bias=True, kernel_initializer='he_normal', return_sequences=True))
#  model.add(LSTM(77, activation='sigmoid', use_bias=True, kernel_initializer='he_normal', return_sequences=True))
# model.add(Dense(77, activation='tanh'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy',
                       'mean_squared_error',  # linear regression performance measure
                       'mean_squared_logarithmic_error',  # used to measure difference between actual and predicted
                       'mean_absolute_error'])  # measure how close predictions are to output])

name = "simple-recurrent-neural-network"  # .format(int(time.time()))

tensorboard = TensorBoard(
    log_dir= # path to where file gets saved
        r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\PRCO304\venv\TensorBoardResults\logs/{}'.format(name), histogram_freq=0,
    write_graph=True)

# callbacks=[tensorboard]

trainModelFit = model.fit(
    trainX,
    trainY,
    epochs=185,
    batch_size=len(trainX),  # Number of samples per gradient update.
    verbose=1,
    callbacks=[tensorboard])

testModelFit = model.fit(
    testX,
    testY,
    epochs=185,
    batch_size=len(testX), #  hard code number
    verbose=1,
    callbacks=[tensorboard])

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print(trainPredict.shape)
print(testPredict.shape)

print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

trainingLoss = trainModelFit.history['loss']
testLoss = testModelFit.history['loss']

trainingAccuracy = trainModelFit.history['acc']
testAccuracy = testModelFit.history['acc']

epochCountLoss = range(1, len(trainingLoss) + 1)
epochCountAccuracy = range(1, len(trainingAccuracy) + 1)

plt.figure(1)
plt.plot(epochCountLoss, trainingLoss, 'r-')
plt.plot(epochCountLoss, testLoss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

plt.figure(2)
plt.plot(epochCountAccuracy, trainingAccuracy, 'r-')
plt.plot(epochCountAccuracy, testAccuracy, 'b-')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show();

# plt.imshow(trainPredict, interpolation='none')
# plt.colorbar()
# plt.show()