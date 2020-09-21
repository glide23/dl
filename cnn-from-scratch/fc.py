# fully-connected layer from scratch in numpy

import numpy as np

def fc_feedforward(x,W,b):
    output = np.matmul(W,x) + b
    return output
