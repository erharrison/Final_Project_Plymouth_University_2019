import tensorflow as tf
from tensorflow.keras import layers
import numpy
numpy.random.seed(0)
import matplotlib.pyplot as plt


dataset = numpy.loadtxt(r"C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\Data.csv", delimiter=",")
inputX, outputY = dataset[:,0:78], dataset[:128]


# Sequential model is a linear stack of layers
model = tf.keras.Sequential()
# 77 features/columns
model.add(layers.Dense(77, input_dim=77, activation='relu'))

model.add(layers.Dense(15, activation='relu'))

model.add(layers.Dense(77 , activation='relu'))


# Compile model
# binary_crossentropy or categorical_crossentropy
# categorical or binary_accuracy
# adam is efficient gradient descent algorithm
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy', 'mse', 'mae', 'mape', 'cosine'])


# Fit the model
# epochs is number of iterations through the dataset
history = model.fit(inputX, outputY, epochs=150, batch_size=len(inputX), verbose=2)

plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['mean_absolute_percentage_error'])
plt.plot(history.history['cosine_proximity'])
plt.show()


# calculate predictions
predictions = model.predict(inputX)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)