import keras
import numpy


class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):  # call makes the forward pass
        prev_output = states[0]
        product = numpy.dot(inputs, self.kernel)
        output = product + numpy.dot(prev_output, self.recurrent_kernel)
        return output, [output]  # (output_at_t, states_at_t_plus_1)
    # output is a tensor. A tensor is a container which can house data in N dimensions, along with its linear operations
