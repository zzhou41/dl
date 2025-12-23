import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

def reshape_mnist(images):
    # images: N x 784
    # return: N x 28 x 28
    return images.reshape(-1, 28, 28)

def image_to_patches(images, patch_size=4):
    # self-attention: (batch, sequence_length, feature_dim)
    # images: (batch, height, width)
    # preserve local spatial meaning
    # 4x4 patches: 28/4=7patches per row, 7x7=49 patches total
    # 1 patch/token: 4x4=16 pixels
    # sequence_length=49, feature_dim=16
    B, H, W = images.shape
    patches = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = images[:, i:i+patch_size, j:j+patch_size]
            # keep B, flatten everything else into one vector per token
            # -1 means figure out the correct size automatically
            patches.append(patch.reshape(B, -1))
    # (B, 49, 16) like sentence of 49 word vectors (flattened letters of pixels)
    return np.stack(patches, axis=1)


#mndata = MNIST('./data')
#images, labels = mndata.load_training()

#print(len(images), len(labels))

#img = np.array(images[0]).reshape(28, 28)
#print(labels[0])
#plt.imshow(img, cmap="gray")
#plt.title(f"Label: {labels[0]}")
#plt.axis("off")
#plt.show()