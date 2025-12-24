import numpy as np
from load import reshape_mnist, image_to_patches
from model import AttentionMNIST
from losses import cross_entropy
from optim import backward
from mnist import MNIST

def train(model, images, labels, epochs=5, batch_size=64, lr=0.01):
    images = reshape_mnist(images)

    for epoch in range(epochs):
        perm = np.random.permutation(len(images))
        images, labels = images[perm], labels[perm]

        total_loss = 0

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i+batch_size]
            batch_lbls = labels[i:i+batch_size]

            x = image_to_patches(batch_imgs)
            logits = model.forward(x)

            loss, probs = cross_entropy(logits, batch_lbls)
            total_loss += loss

            backward(model, probs, batch_lbls, lr)

        avg_loss = total_loss / (len(images) // batch_size)
        print(f"Epoch {epoch+1}, loss: {avg_loss:.4f}")


def main():
    mndata = MNIST('./data')
    images, labels = mndata.load_training()
    # normalize pixel values to [0, 1]
    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int64)

    model = AttentionMNIST()

    train(
        model,
        images,
        labels,
        epochs=15,
        batch_size=64,
        lr=0.05
    )

if __name__ == "__main__":
    main()