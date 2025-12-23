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
    
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, dL):
        for layer in reversed(self.layers):
            dL = layer.backward(dL)

    def binary_cross_entropy(self, y_hat, y):
        eps = 1*(10)**(-8)
        return np.mean(-(y*np.log(y_hat+eps) + (1-y) * np.log(1-y_hat+eps)))

    def train(self, X, y, epochs=1000):
        n = X.shape[0]

        for epoch in range(epochs):
            y_hat = self.forward(X)
            loss = self.binary_cross_entropy(y_hat, y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} completed. BCE Loss: {loss:.4f}")

            dL = (y_hat - y) / n
            self.backward(dL)

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n = 100
    d = 10
    h = 30
    o = 1

    X = np.random.randn(n, d)
    W = np.random.randn(d, 1)
    y = (np.dot(X, W) > 0).astype(int) 

    model = NeuralNetwork([
        Linear(d, h, learning_rate=0.1),
        ReLU(),
        Linear(h, o, learning_rate=0.1),
        Sigmoid()
    ])

    model.train(X, y, epochs=4000)
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Training accuracy: {accuracy * 100:.2f}%")
