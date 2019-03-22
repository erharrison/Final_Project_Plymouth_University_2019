import pandas
import keras
import matplotlib
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, LSTMCell
from sklearn.preprocessing import MinMaxScaler
import numpy
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import time
from tensorflow.python.ops import rnn, rnn_cell

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv(
    r"C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\Data.csv",
    delimiter=",",
    engine='python')

dataset = dataframe.values
# floating point values are more suitable for neural networks
dataset = dataset.astype('float32')

# normalize the dataset - batch normalisation???
scaler = MinMaxScaler(feature_range=(0, 1))  # MinMaxScalar is from ssikit learn library
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.15)  # 15%
test_size = int(len(dataset) * 0.15)  # 15%
train, test = dataset[0:train_size, :], dataset[train_size:(2*train_size), :]
print(len(train), len(test))

trainX, trainY = train[1:-1, :], train[2:, :]
testX, testY = test[1:-1, :], test[2:, :]

# (batch_size, timesteps, features)
# trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
# trainY = trainY.reshape(trainY.shape[0], 1, trainY.shape[1])
# testX = testX.reshape(testX.shape[0], 1, testX.shape[1])
# testY = testY.reshape(testY.shape[0], 1, testY.shape[1])

lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)

# create RNN model
# Sequential model is a linear stack of layers
model = Sequential()
model.add(keras.layers.RNN(LSTMCell(77).state_size(call=testX, state_is_tuple=True), activation='relu', use_bias=True, kernel_initializer='he_normal', return_sequences=True))
model.add()
#  model.add(keras.layers.LSTM(77, activation='relu', use_bias=True, kernel_initializer='he_normal', return_sequences=True))
#  model.add(LSTM(77, activation='relu', use_bias=True, kernel_initializer='he_normal', return_sequences=True))
#  model.add(Dense(77, activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy',
                       'mean_squared_error',  # linear regression performance measure
                       'mean_squared_logarithmic_error',  # used to measure difference between actual and predicted
                       'mean_absolute_error'])  # measure how close predictions are to output])

name = "recurrent-neural-network-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='/TensorBoardResults/logs/{}'.format(name), histogram_freq=0,
                          write_graph=True)

model.fit(trainX, trainY, epochs=1000, batch_size=len(trainX), verbose=2, callbacks=[tensorboard])
model.fit(testX, testY, epochs=1000, batch_size=len(testX), verbose=2, callbacks=[tensorboard])

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print(trainPredict.shape)
print(testPredict.shape)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(numpy.arange(data.shape[1]))
    ax.set_yticks(numpy.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(numpy.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(numpy.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["green", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, numpy.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

fig, ax = plt.subplots()


im, cbar = heatmap(trainPredict, len(trainPredict), 'len(trainPredict[0])', ax=ax,
                   cmap="YlGn", cbarlabel=" no. of infected samples")
texts = annotate_heatmap(im, valfmt="{x:.1f} t")

fig.tight_layout()
plt.show()