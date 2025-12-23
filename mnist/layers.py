import numpy as np
from utils import softmax

class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) *0.02

    