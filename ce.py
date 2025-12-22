import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim, learning_rate = 0.01):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        self.learning_rate = learning_rate

    def forward(self, X):
        self.X = X  # n x input_dim
        return X @ self.W + self.b  # n x output_dim
    
    def backward(self, dZ):  # dZ: n x output_dim
        dW = self.X.T @ dZ  # input_dim x output_dim
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = dZ @ self.W.T  # n x input_dim

        self.W -= dW * self.learning_rate
        self.b -= db * self.learning_rate
        return dX
    
class ReLU:
    def forward(self, Z):
        self.Z = Z
        return np.maximum(Z, 0)
    
    def backward(self, dA):
        return dA * (self.Z > 0)
    
class Sigmoid:
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A
    
    def backward(self, dA):
        return dA  # binary cross entropy + sigmoid

class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        self.n = logits.shape[0]
        # stablize
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        self.y_hat = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        if y.ndim == 1:
            self.y = np.zeros_like(self.y_hat)
            self.y[np.arange(self.n), y] = 1
        else:
            self.y = y

        eps = 10**(-8)
        loss = -np.sum(self.y * np.log(self.y_hat + eps)) / self.n
        return loss
    
    def backward(self):
        return (self.y_hat - self.y) / self.n

class NeuralNetwork:
    def __init__(self, layers, loss_fn):
        self.layers = layers
        self.loss_fn = loss_fn

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, dL):
        for layer in reversed(self.layers):
            dL = layer.backward(dL)

    def train(self, X, y, epochs=1000):
        n = X.shape[0]

        for epoch in range(epochs):
            y_hat = self.forward(X)
            loss = self.loss_fn.forward(y_hat, y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} completed. CE Loss: {loss:.4f}")

            dL = self.loss_fn.backward()
            self.backward(dL)

    def predict(self, X):
        logits = self.forward(X)
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        y_hat = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return np.argmax(y_hat, axis=1)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n = 100
    d = 2
    k = 3

    X = np.random.randn(n, d)
    # y = np.random.randint(0, k, size=n)  # no learnable features in dataset
    y = np.zeros(n, dtype=int)
    y[X[:, 0] > 0] = 1
    y[X[:, 1] > 0] = 2

    model = NeuralNetwork(
        layers=[Linear(d, k, learning_rate=0.8)],
        loss_fn=SoftmaxCrossEntropy())

    model.train(X, y, epochs=8000)
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Training accuracy: {accuracy * 100:.2f}%")
