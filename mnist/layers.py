import numpy as np
from utils import softmax

class Linear:
    def __init__(self, in_dim, out_dim):
        # std around 0.02 safe conservative default in transformers
        self.W = np.random.randn(in_dim, out_dim) *0.02

    def forward(self, x):
        self.x = x
        return x @ self.W
    
    def backward(self, grad, lr):
        # (B, N, in_dim) -> (BxN, in_dim) flatten all batch dimensions
        # (in_dim, BxN) @ (BxN, out_dim) -> (in_dim, out_dim)
        dW = self.x.reshape(-1, self.x.shape[-1]).T @ grad.reshape(-1, grad.shape[-1])
        self.W -= lr * dW
        return grad @ self.W.T

class SelfAttention:
    def __init__(self, dim):
        self.Wq = np.random.randn(dim, dim) * 0.02
        self.Wk = np.random.randn(dim, dim) * 0.02
        self.Wv = np.random.randn(dim, dim) * 0.02

    def forward(self, x):
        self.x = x  # (B, N, D)
        self.Q = x @ self.Wq  # (B, number of patches, embedding size)
        self.K = x @ self.Wk  # (B, number of patches, embedding size)
        self.V = x @ self.Wv
        # (B, N, D) @ (B, D, N) --> (B, N, N)
        # each patch attends to every other patch
        # sqrt(D): keep softmax smooth from saturation, trainable, numerically stable
        scores = self.Q @ self.K.transpose(0, 2, 1) / np.sqrt(x.shape[-1])
        self.attn = softmax(scores, axis=-1)  # for each token
        return self.attn @ self.V  # (B, N, D)
        # each output token is a weighted sum of all value vectors
        
    def backward(self, grad, lr):
        B, N, D = grad.shape
        dV = self.attn.transpose(0, 2, 1) @ grad
        dattn = grad @ self.V.transpose(0, 2, 1)

        # softmax backward
        # Jacobian: diag(a) - aa.T
        dscores = self.attn * (
            dattn - (dattn * self.attn).sum(axis=-1, keepdims=True)
        )
        dscores /= np.sqrt(D)

        dQ = dscores @ self.K
        dK = dscores.transpose(0, 2, 1) @ self.Q

        x_flat = self.x.reshape(-1, D)
        self.Wq -= lr * (x_flat.T @ dQ.reshape(-1, D))
        self.Wk -= lr * (x_flat.T @ dK.reshape(-1, D))
        self.Wv -= lr * (x_flat.T @ dV.reshape(-1, D))

        dx = (
            dQ @ self.Wq.T +
            dK @ self.Wk.T +
            dV @ self.Wv.T
        )

        return dx