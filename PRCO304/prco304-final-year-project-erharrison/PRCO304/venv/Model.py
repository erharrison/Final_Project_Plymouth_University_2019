import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy
import matplotlib.pyplot as plt

seed = 10
numpy.random.seed(seed)  # for reproducibility

scaler = StandardScaler()  # scaling beacuse mlp is sensitive to feature scaling

dataset = numpy.loadtxt(r"C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\Data.csv", delimiter=",")
inputX, outputY = dataset[:,0:78], dataset[:128]

inputX_train, inputX_test, outputY_train, inputY_test = train_test_split(inputX, outputY)

scaler.fit(inputX_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

inputX_train = scaler.transform(inputX_train)
inputX_test = scaler.transform(inputX_test)

outputY_train = scaler.transform(outputY_train)
inputY_test = scaler.transform(inputY_test)

# Sequential model is a linear stack of layers
model = tf.keras.Sequential()
# 77 features/columns
# originally using RELU but changed to sigmoid because loss was negative with other activation functions
model.add(layers.Dense(77, input_dim=77, activation='sigmoid'))

model.add(layers.Dense(15, activation='sigmoid'))

model.add(layers.Dense(77, activation='sigmoid'))


# Compile model
# binary_crossentropy or categorical_crossentropy
# categorical or binary_accuracy
# adam is efficient gradient descent algorithm
model.compile(loss='binary_crossentropy',   # crossentropy measures the divergence between two probability distribution
              optimizer='adam',
              metrics=['accuracy',
                       'categorical_accuracy',
                       'binary_accuracy',
                       'mean_squared_error',    # linear regression performance measure
                       'mean_squared_logarithmic_error',     # used to measure difference between actual and predicted
                       'mean_absolute_error'    # measure how close predictions are to output
                       ])


# Fit the model
# epochs is number of iterations through the training set
# batch size is number of training instances observed before the optimizer performs a
# weight update
modelFit = model.fit(inputX_train, outputY_train, epochs=200, batch_size=len(inputX_train), verbose=1)


#TODO need to use weights, bias input somehow in prediction?

# calculate predictions
predictions = model.predict(inputX_train)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

# plot metrics
plt.figure()
plt.plot((modelFit.history['mean_squared_error']), 'b', label='Mean Squared Error')
plt.plot((modelFit.history['mean_squared_logarithmic_error']), 'r', label='Mean Squared Logarithmic Error')
plt.plot((modelFit.history['mean_absolute_error']), 'y', label='Mean Absolute Error')
plt.title('Regression Metrics')
plt.legend()