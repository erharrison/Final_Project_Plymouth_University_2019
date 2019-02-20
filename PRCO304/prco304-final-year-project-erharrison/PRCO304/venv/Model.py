import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
import numpy
numpy.random.seed(0)


dataset = numpy.loadtxt(r"C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\Data.csv", delimiter=",")
inputX = dataset[:,0:78]
outputY = dataset[:128]


# Sequential model is a linear stack of layers
model = tf.keras.Sequential()
# 77 features/columns
model.add(layers.Dense(77, input_dim=77, activation='relu'))
# second hidden layer had 8 neurons
model.add(layers.Dense(77, activation='relu'))
# ouput layer has 1 neuron to predict class (onset of diabetes or not)
model.add(layers.Dense(77 , activation='sigmoid'))


# Compile model
# binary_crossentropy is logarithmic loss
# adam is efficient gradient descent algorithm
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
# epochs is number of iterations through the dataset
model.fit(inputX, outputY, epochs=150, batch_size=10)


# calculate predictions
predictions = model.predict(inputX)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)