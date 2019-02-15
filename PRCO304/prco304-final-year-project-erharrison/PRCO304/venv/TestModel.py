# # loading data
# import tensorflow as tf
# from tensorflow.keras import layers
#
# model = tf.keras.Sequential()
# # Dense defines fully connected layers
# # Adds a densely-connected layer with 64 units to the model:
# model.add(layers.Dense(64, activation='relu'))
# # Add another:
# model.add(layers.Dense(64, activation='relu'))
# # Add a softmax layer with 10 output units:
# model.add(layers.Dense(10, activation='softmax'))


from keras.models import Sequential
from keras.layers import Dense
import numpy
numpy.random.seed(7)

dataset = numpy.loadtxt(r"C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

# ------

# create model
model = Sequential()
# first layer has 12 neurons and input_dim is how many inputs in first layer. relu is the activation function on the first two layers and the sigmoid function in the output layer
model.add(Dense(12, input_dim=8, activation='relu'))
# second hidden layer had 8 neurons
model.add(Dense(8, activation='relu'))
# ouput layer has 1 neuron to predict class (onset of diabetes or not)
model.add(Dense(1, activation='sigmoid'))

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