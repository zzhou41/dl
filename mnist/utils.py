import numpy as np

def softmax(x, axis=1):
    # stablize
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def one_hot(y, num_classes):
    # y: (n, ) --> one_hot: (n, num_classes)
    output = np.zeros([y.shape[0], num_classes])
    output[np.arange(y.shape[0]), y] = 1
    return output