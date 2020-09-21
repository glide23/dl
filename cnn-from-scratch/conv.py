# convolutional layer from scratch using numpy

import numpy as np

def conv_feedforward(x,kernel):
    k_height = kernel.shape[0]
    k_width = kernel.shape[1]
    x_height = x.shape[0]
    x_width = x.shape[1]
    out_height = x_height-k_height+1
    out_width = x_width-k_width+1
    output = np.zeros((out_height,out_width))
    i,j = 0,0
    for i in range(out_height):
        for j in range(out_width):
            window_curr = x[i:i+k_height, j:j+k_width]
            window_result = kernel * window_curr
            window_sum = np.sum(np.sum(window_result))
            output[i,j] = window_sum
    return output