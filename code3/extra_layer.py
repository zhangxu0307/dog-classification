from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from numpy import newaxis

def outer_product(inputs):
    """
    inputs: list of two tensors (of equal dimensions,
        for which you need to compute the outer product
    """
    x, y = inputs
    batchSize = K.shape(x)[0]
    outerProduct = x[:,:, newaxis] * y[:,newaxis,:]
    outerProduct = K.reshape(outerProduct, (batchSize, -1))
    # returns a flattened batch-wise set of tensors
    return outerProduct

class MyLayer(Layer):

    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):

        x = inputs[0]
        y = inputs[1]
        print(K.int_shape(x))
        print(K.int_shape(y))

        x = K.permute_dimensions(x, [0, 3, 1, 2])
        y = K.permute_dimensions(y, [0, 3, 1, 2])
        print(K.int_shape(x))
        print(K.int_shape(y))

        w = K.int_shape(x)[2]
        h = K.int_shape(x)[3]
        c = K.int_shape(x)[1]
        self.c = c
        x = K.reshape(x, [-1, c, w * h])
        y = K.reshape(y, [-1, c, w * h])
        print(K.int_shape(x))
        print(K.int_shape(y))

        x_T = K.permute_dimensions(x, [0, 2, 1])
        # print(K.int_shape(x))
        # print(K.int_shape(y))
        phi_I = K.batch_dot(x_T, y, axes=(1, 2))

        #phi_I = outer_product(inputs=[x, y])
        print(K.int_shape(phi_I))

        phi_I = K.reshape(phi_I, [-1, c * c])
        print(K.int_shape(phi_I))

        phi_I = phi_I / 784.0

        y_ssqrt = K.sign(phi_I) * K.sqrt(K.abs(phi_I) + 1e-12)
        z_l2 = K.l2_normalize(y_ssqrt, axis=1)

        return z_l2

    def compute_output_shape(self, input_shape):
        self.output_dim = self.c*self.c
        print(input_shape[0][0])
        return (input_shape[0][0], self.output_dim)
