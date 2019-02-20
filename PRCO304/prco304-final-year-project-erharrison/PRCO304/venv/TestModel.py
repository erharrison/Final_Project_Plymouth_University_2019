import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
import numpy
numpy.random.seed(7)

dataset = numpy.loadtxt(r"C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

# ------

# create model
model = tf.keras.Sequential()
# first layer has 12 neurons and input_dim is how many inputs in first layer. relu is the activation function on the first two layers and the sigmoid function in the output layer
model.add(layers.Dense(12, input_dim=8, activation='relu'))
# second hidden layer had 8 neurons
model.add(layers.Dense(8, activation='relu'))
# ouput layer has 1 neuron to predict class (onset of diabetes or not)
model.add(layers.Dense(1, activation='sigmoid'))

# -----

# Compile model
# binary_crossentropy is logarithmic loss
# adam is efficient gradient descent algorithm
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# -----

# Fit the model
# epochs is number of iterations through the dataset
model.fit(X, Y, epochs=150, batch_size=10)

# -----

# evaluate model
# scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# -----

# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)