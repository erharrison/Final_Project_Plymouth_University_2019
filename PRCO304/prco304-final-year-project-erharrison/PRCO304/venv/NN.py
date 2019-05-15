# fix random seed for reproducibility
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout
from sklearn.preprocessing import MinMaxScaler as mms
import folium as fl
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard as tb
from keras.utils.vis_utils import plot_model
import os  # for ghraphviz


# adding graphviz to the PATH
os.environ["PATH"] += os.pathsep + r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\graphviz-2.38\release\bin'

# addresses to file paths
coordinates_file_path = r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\Coordinates.xlsx'
data_file_path = r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\ImputedData.xlsx'
map_path = r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\mymap.html'
tensorboard_path = r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\PRCO304\venv\TensorBoardResults\logs\{}'

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
                    activation='linear'))
model.add(Dropout(0.1))  # adding 10% dropout to neurons of the hidden layer
model.add(Dense(77,
                activation='linear'))

# compiling model
model.compile(loss='mean_squared_error',
              optimizer='Nadam',
              metrics=['accuracy',
                       'mean_squared_error',
                       'mean_squared_logarithmic_error',
                       'mean_absolute_error'])


# setting up tensorboard
name = "simple-recurrent-neural-network"
tensorboard = tb(
    log_dir=
    tensorboard_path.format(name),  # path to where file gets saved
    histogram_freq=0,
    write_graph=True)

# fitting the model
trainModelFit = model.fit(
    trainX,
    trainY,
    validation_data=(testX, testY),
    epochs=200,
    batch_size=1,  # Number of samples per gradient update.
    verbose=1,
    callbacks=[tensorboard])  # running tensorboard

# calculating accuracy score
scores = model.evaluate(testX, testY, verbose=0)  # score of accuracy
print('accuracy score = {}'.format(scores[0]))


# make predictions
trainPredict = model.predict(trainX)


# Visualising model in python console
print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

trainingLoss = trainModelFit.history['loss']
trainingAccuracy = trainModelFit.history['acc']

epochCountLoss = range(1, len(trainingLoss) + 1)
epochCountAccuracy = range(1, len(trainingAccuracy) + 1)

# setting the axes for the loss graph, so that it is the same as the graph of metrics
axes = plt.gca()
axes.set_xlim([0, 200])
axes.set_ylim([0, 0.06])

# creating matplotlib graph for plotting loss
plt.figure(1)
plt.plot(epochCountLoss, trainingLoss, 'b')
plt.title('Rate of Loss')
plt.legend(['Training Loss',])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

# creating matplotlib graph for plotting accuracy
plt.figure(2)
plt.plot(epochCountAccuracy, trainingAccuracy, 'r-')
plt.title('Accuracy of Training')
plt.legend(['Training Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show();

# plot metrics
plt.figure(3)
plt.plot((trainModelFit.history['mean_squared_error']), 'b')
plt.plot((trainModelFit.history['mean_squared_logarithmic_error']), 'r')
plt.plot((trainModelFit.history['mean_absolute_error']), 'y')
plt.title('Regression Metrics')
plt.legend(['Mean Squared Error', 'Mean Squared Logarithmic Error', 'Mean Absolute Error'])
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
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

# Adding circle one by one on the map
for i in range(0, len(dataset[0])):  # for loop going for the length of the dataset
    if dataset[year1, i] > 0:  # iterating through the locations (columns) of the year the user input
        fl.Circle(
            # iterating through the coordinates from the Excel file to get longitude and latitude coordinates
            location=[coordinates.iloc[i]['lon'], coordinates.iloc[i]['lat']],
            radius=(dataset[year1, i]) * 1000000,
            color='#99CCFF',  # css colour code for blue
            fill=True,
            fill_color='#99CCFF'
        ).add_to(predictions_map)

for i in range(0, len(trainPredict[0, 0])):
    # [sheet, row, column] iterating throught the columns of the first row of the sheet corresponding to the chosen year
    if trainPredict[year2, 0, i] > 0:  # leaving out negative predictions
        fl.Circle(
            location=[coordinates.iloc[i]['lon'], coordinates.iloc[i]['lat']],
            radius=(trainPredict[year2, 0, i]) * 1000000,
            color='#DC143C',  # css colour code for crimson
            fill=True,
            fill_color='#DC143C'
        ).add_to(predictions_map)

# Save it as html
predictions_map.save(map_path)
