import numpy as np
from utils import one_hot

def backward(model, probs, labels, lr):
    B = labels.shape[0]
    y = one_hot(labels, 10)
    dlogits = (probs - y) / B
    dpooled = model.classifier.backward(dlogits, lr)
    # every patch receives an equal share of the gradient
    N = model.x.shape[1]
    dpatches = np.repeat(dpooled[:, None, :], N, axis=1) / N  # (B, 49, 64)
    dattn = model.attn.backward(dpatches, lr)
    dx = dpatches + dattn  # residual
    model.embed.backward(dx, lr)