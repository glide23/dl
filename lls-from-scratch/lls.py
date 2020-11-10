import numpy as np

# Linear Least Squares method implemented in numpy, for *invertible* X matrices only
# input:
#   X - a matrix which rows hold our data's samples
#   y_true - a vector which cells hold the groundtruth value for each sample
# output:
#   the weights vector for each dimension of the input
def lls(X, y_true):
    X = np.array(X)
    y_true = np.array(y_true)
    W = np.linalg.inv(np.matmul(X.transpose(), X))
    W = np.matmul(W,X.transpose())
    W = np.matmul(W,y_true)
    return W

# sample code
X = [[1,3],[2,5]]
y_true = [5,9]
print("calculated weights:",lls(X,y_true))
