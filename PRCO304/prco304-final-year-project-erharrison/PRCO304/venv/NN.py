import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, LSTMCell, Activation, RNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.callbacks import TensorBoard
import time
from keras.utils.vis_utils import plot_model
import os  # for ghraphviz
import cv2

#  pd.options.mode.chained_assignment = None  # default='warn'

# fix random seed for reproducibility
seed = 0
numpy.random.seed(seed)

# adding graphviz to the PATH
os.environ["PATH"] += os.pathsep + 'C:/Users/emily/Downloads/graphviz-2.38/release/bin'

# creating image for map of North America for data to be visualised on
map_img = mpimg.imread(
    r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\Map.jpg')

# Reading Excel file and spreadsheet of original data
data = pd.read_excel(
    r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\OriginalData.xlsx', sheet_name='Samples')

# Creating dataframe from data and selecting columns
dataframe = pd.DataFrame(data,
                         columns=['decimalLongitude',
                                  'decimalLatitude',
                                  'diseaseDetected',
                                  'year',
                                  'country'])


# Creating a dictionary file for countries
country = {'USA': 0, 'Canada': 1, 'Mexico': 2}
# Replacing each item in country column with number, according to dictionary
dataframe['country'] = [country[item] for item in dataframe['country']]

for i in range(len(dataframe['decimalLongitude'])):
    longitude = dataframe['decimalLongitude']
    # Replacing each item in loongitude column with number, according to dictionary
    longitude[i] = int(round(longitude[i]))

# Replacing dataframe with copy
dataframe['decimalLongitude'] = longitude


# Specifying a writer
writer = pd.ExcelWriter(r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\NewData.xlsx', engine='xlsxwriter')

# Writing dataframe to new Excel file
dataframe.to_excel(excel_writer=writer, sheet_name='Data', header=False)
writer.save()

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
testX, testY = test[1:-1, :], test[2:, :] # shape is 20


# (batch_size, timesteps, features)
trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])  # trainX use fisrt row and see how that goes then use 20 rows etc.
trainY = trainY.reshape(trainY.shape[0], 1, trainY.shape[1])
testX = testX.reshape(testX.shape[0], 1, testX.shape[1])
testY = testY.reshape(testY.shape[0], 1, testY.shape[1])
print(trainX.shape, testX.shape)


# create recurrent neural network
model = Sequential()  # Sequential model is a linear stack of layers
#  model.add(Activation('relu'))
#  model.add(RNN(cell, return_sequences=True))
model.add(keras.layers.SimpleRNN(77, return_sequences=True))
#  model.add(LSTM(77, activation='sigmoid', use_bias=True, kernel_initializer='he_normal', return_sequences=True))
model.add(Dense(77, activation='tanh'))

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

trainModelFit = model.fit(
    trainX,
    trainY,
    validation_data=(testX,testY),
    epochs=100,
    batch_size=1,  # Number of samples per gradient update.
    verbose=1,
    callbacks=[tensorboard])

scores = model.evaluate(testX,testY,verbose=0)  # score of accuracy
print(scores[0])

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print(trainPredict.shape)
print(testPredict.shape)

# TODO iterate through predictions
# TODO circle needs to be in location of long+lat
# Creating circles on map to visualise predictions
cv2.circle(map_img, (200, 200), trainPredict[0, 0, 0], (0, 20, 200), 2)
cv2.imshow('Map', map_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Visualising model
print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

trainingLoss = trainModelFit.history['loss']
trainingAccuracy = trainModelFit.history['acc']

epochCountLoss = range(1, len(trainingLoss) + 1)
epochCountAccuracy = range(1, len(trainingAccuracy) + 1)

plt.figure(1)
plt.plot(epochCountLoss, trainingLoss, 'r-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

plt.figure(2)
plt.plot(epochCountAccuracy, trainingAccuracy, 'r-')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show();