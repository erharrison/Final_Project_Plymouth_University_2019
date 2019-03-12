import numpy
import pandas
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv(
    r"C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\
    prco304-final-year-project-erharrison\Data.csv",
    delimiter=",",
    engine='python')

dataset = dataframe.values
# floating point values are more suitable for neural networks
dataset = dataset.astype('float32')

inputX, outputY = dataset[1:-1, :], dataset[2:, :]