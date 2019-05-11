# fix random seed for reproducibility
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler as mms
import folium as fl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard as tb
from keras import regularizers
from keras.utils.vis_utils import plot_model
import os  # for ghraphviz
import datetime


# adding graphviz to the PATH
os.environ["PATH"] += os.pathsep + r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\graphviz-2.38\release\bin'

# addresses to file path
coordinates_file_path = r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\Coordinates.xlsx'
data_file_path = r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\ImputedData.xlsx'
map_path = r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\mymap.html'


# Reading Excel file and spreadsheet of original data
data_file_excel = pd.read_excel(data_file_path, sheet_name='Data', header=None)
# Creating dataframe from data
dataframe_excel = pd.DataFrame(data_file_excel)
dataset_excel = dataframe_excel.values
# floating point values are more suitable for neural networks
dataset_excel = dataset_excel.astype('float32')

scalar = mms(feature_range=(0, 1))
dataset = scalar.fit_transform(dataset_excel)  # scaling the dataset to be between 0 and 1

# split into train and test sets
train_size = int(len(dataset) * 0.8)  # 80%
test_size = int(len(dataset) * 0.2)  # 20%
train, test = dataset[0:train_size, :], dataset[train_size:(2*train_size), :]

# train and test datasets separated into input and expected output
trainX, trainY = train[1:-1, :], train[2:, :]
testX, testY = test[1:-1, :], test[2:, :]


# reshape to (batch_size, timesteps, features) for input for RNN
trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
trainY = trainY.reshape(trainY.shape[0], 1, trainY.shape[1])
testX = testX.reshape(testX.shape[0], 1, testX.shape[1])
testY = testY.reshape(testY.shape[0], 1, testY.shape[1])


# create recurrent neural network
model = Sequential()  # Sequential model is a linear stack of layers
model.add(SimpleRNN(77,
                    return_sequences=True,
                    activation='linear',
                    kernel_initializer='glorot_normal'))
model.add(Dense(77,
                activation='linear'
                )
          )

model.compile(loss='mean_squared_error',
              optimizer='Nadam',
              metrics=['accuracy',
                       'mean_squared_error',  # linear regression performance measure
                       'mean_squared_logarithmic_error',  # used to measure difference between actual and predicted
                       'mean_absolute_error'])  # measure how close predictions are to output])

time = datetime.datetime.now().time()
name = "simple-recurrent-neural-network-%s" % time

tensorboard = tb(
    log_dir=  # path to where file gets saved
        r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\PRCO304\venv\TensorBoardResults\logs\{}'.format(name), histogram_freq=0,
    write_graph=True)

trainModelFit = model.fit(
    trainX,
    trainY,
    validation_data=(testX,testY),
    epochs=100,
    batch_size=1,  # Number of samples per gradient update.
    verbose=1,
    callbacks=[tensorboard])

scores = model.evaluate(testX, testY, verbose=0)  # score of accuracy
print(scores[0])


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Visualising model
print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

trainingLoss = trainModelFit.history['loss']
trainingAccuracy = trainModelFit.history['acc']

epochCountLoss = range(1, len(trainingLoss) + 1)
epochCountAccuracy = range(1, len(trainingAccuracy) + 1)

# creating matplotlib graph for plotting loss
plt.figure(1)
plt.plot(epochCountLoss, trainingLoss, 'r-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

# creating matplotlib graph for plotting accuracy
plt.figure(2)
plt.plot(epochCountAccuracy, trainingAccuracy, 'r-')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show();

# reading coordinates Excel file and creating Dataframe
coordinates_file = pd.read_excel(coordinates_file_path, sheet_name='Coordinates', header=None)
dataframe_coordinates = pd.DataFrame(coordinates_file)
coordinates = dataframe_coordinates.values
coordinates = coordinates.astype('float32')

coordinates = pd.DataFrame({
   'lat': coordinates_file.iloc[0],  # this gets first row - latitude
   'lon': coordinates_file.iloc[1],  # gets first second - longitude
})

# Making an empty folium map
predictions_map = fl.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2)

year1 = int(input("What year from the dataset do you want to map? (between 1888-2014)"))
# to get row number from input year
year1 = year1-1888

year2 = int(input("How many years into the future do you want to map for comparison? (max. 99 years)"))
year2 = year2-1  # a regular user is unlikely to assume zero based numbering

for i in range(0, len(dataset[0])):
    if dataset[year1, i] > 0:
        fl.Circle(
            location=[coordinates.iloc[i]['lon'], coordinates.iloc[i]['lat']],
            popup=str(dataset[year1, i]),
            radius=(dataset[year1, i]) * 100000,
            color='#99CCFF',
            fill=True,
            fill_color='#99CCFF'
        ).add_to(predictions_map)

# I can add marker one by one on the map
for i in range(0, len(trainPredict[0, 0])):
    # [sheet, row, column]
    if trainPredict[year2, 0, i] > 0:  # leaving out negative predictions
        fl.Circle(
            location=[coordinates.iloc[i]['lon'], coordinates.iloc[i]['lat']],
            popup=str(trainPredict[year2, 0, i]),
            radius=(trainPredict[year2, 0, i]) * 1000000,  # TODO need to figure out what to do about negative predictions
            color='#DC143C',
            fill=True,
            fill_color='#DC143C'
        ).add_to(predictions_map)


# Save it as html
predictions_map.save(map_path)
