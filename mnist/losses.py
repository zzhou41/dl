from utils import softmax
import numpy as np

def cross_entropy(logits, labels):
    # (batch_size, num_classes)
    probs = softmax(logits, axis=1)
    # labels: (B,)
    # picks [probs[0, labels[0]], ...]
    # model's confidence in the correct class for each sample
    # mean: normalize across batch size
    loss = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 10**(-9)))
    return loss, probs