# TODO Data augmentation

import pandas
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
import numpy
import folium
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.callbacks import TensorBoard
from keras.utils.vis_utils import plot_model
import os  # for ghraphviz

# fix random seed for reproducibility
seed = 0
numpy.random.seed(seed)

# adding graphviz to the PATH
os.environ["PATH"] += os.pathsep + r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\graphviz-2.38\release\bin'

coordinates_file_path = r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\Coordinates.xlsx'
data_file_path_excel = r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\ImputedData.xlsx'

# Reading Excel file and spreadsheet of original data
data_file_excel = pandas.read_excel(data_file_path_excel, sheet_name='Data', header=None)
# Creating dataframe from data and selecting columns
dataframe_excel = pandas.DataFrame(data_file_excel)
dataset_excel = dataframe_excel.values
# floating point values are more suitable for neural networks
dataset_excel = dataset_excel.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))  # MinMaxScalar is from scikit learn library
dataset = scaler.fit_transform(dataset_excel)

# split into train and test sets
train_size = int(len(dataset) * 0.8)  # 80% - train_size = 84
test_size = int(len(dataset) * 0.2)  # 20% - test_size = 21
train, test = dataset[0:train_size, :], dataset[train_size:(2*train_size), :]
print(len(train), len(test))

# TODO experiment with different test datasets
trainX, trainY = train[1:-1, :], train[2:, :]  # trainX shape is 82
testX, testY = test[1:-1, :], test[2:, :]  # testX shape is 20


# (batch_size, timesteps, features)
trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
trainY = trainY.reshape(trainY.shape[0], 1, trainY.shape[1])
testX = testX.reshape(testX.shape[0], 1, testX.shape[1])
testY = testY.reshape(testY.shape[0], 1, testY.shape[1])
print(trainX.shape, testX.shape)


# create recurrent neural network
model = Sequential()  # Sequential model is a linear stack of layers
model.add(SimpleRNN(77, return_sequences=True, activation='linear', kernel_initializer='glorot_normal'))
model.add(Dense(77, activation='linear'))

model.compile(loss='mean_squared_error',
              optimizer='Nadam',
              metrics=['accuracy',
                       'mean_squared_error',  # linear regression performance measure
                       'mean_squared_logarithmic_error',  # used to measure difference between actual and predicted
                       'mean_absolute_error'])  # measure how close predictions are to output])

name = "simple-recurrent-neural-network"  # .format(int(time.time()))

tensorboard = TensorBoard(
    log_dir= # path to where file gets saved
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

print(trainPredict.shape)
print(testPredict.shape)

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

coordinates_file = pandas.read_excel(coordinates_file_path, sheet_name='Coordinates', header=None)
dataframe_coordinates = pandas.DataFrame(coordinates_file)
coordinates = dataframe_coordinates.values
coordinates = coordinates.astype('float32')

coordinates = pandas.DataFrame({
   'lat': coordinates_file.iloc[0],  # this gets first second row - latitude
   'lon': coordinates_file.iloc[1],  # gets first row - longitude
})

# Making an empty folium map
predictions_map = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2)

# transpose prediction array
# trainPredict = numpy.ndarray.transpose(trainPredict)

year1 = int(input("What year from the dataset do you want to map? (between 1888-2014)"))
# to get row number from input year
year1 = year1-1888

year2 = int(input("How many years into the future do you want to map for comparison? (max. 98 years)"))

for i in range(0, len(dataset[0])):
    if dataset[year1, i] > 0:
        folium.Circle(
            location=[coordinates.iloc[i]['lon'], coordinates.iloc[i]['lat']],
            popup=str(dataset[year1, i]),
            radius=(dataset[year1, i]) * 1000000,
            color='#99ccff',
            fill=True,
            fill_color='#99ccff'
        ).add_to(predictions_map)

# I can add marker one by one on the map
for i in range(0, len(trainPredict[0, 0])):
    # [sheet, row, column]
    if trainPredict[year2, 0, i] > 0:  # leaving out negative predictions
        folium.Circle(
            location=[coordinates.iloc[i]['lon'], coordinates.iloc[i]['lat']],
            popup=str(trainPredict[year2, 0, i]),
            radius=(trainPredict[year2, 0, i]) * 1000000,  # TODO need to figure out what to do about negative predictions
            color='crimson',
            fill=True,
            fill_color='crimson'
        ).add_to(predictions_map)


# Save it as html
predictions_map.save(
    r'C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\mymap.html')
