import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(int)

    def forward(self, X):
        # Forward propagation
        self.Z1 = X @ self.W1 + self.b1  # n x hidden_size
        self.A1 = self.relu(self.Z1)  # n x hidden_size
        self.Z2 = self.A1 @ self.W2 + self.b2  # n x output_size
        self.A2 = self.sigmoid(self.Z2)  # n x output_size
        return self.A2

    def backward(self, X, y):
        # Backward propagation
        n = X.shape[0]
        
        dA2 = (self.A2 - y) / n  # n x output_size
        dZ2 = dA2 * self.sigmoid_derivative(self.A2)  # n x output_size
        dW2 = self.A1.T @ dZ2  # hidden_size x output_size
        db2 = np.sum(dZ2, axis=0, keepdims=True)  # 1 x output_size

        dA1 = dZ2 @ self.W2.T  # n x hidden_size
        dZ1 = dA1 * self.relu_derivative(self.Z1)  # n x hidden_size
        dW1 = X.T @ dZ1  # input_size x hidden_size
        db1 = np.sum(dZ1, axis=0, keepdims=True)  # 1 x hidden_size
        
        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            self.forward(X)
            if epoch % 100 == 0:
                loss = np.mean((self.A2 - y) ** 2)
                print(f"Epoch {epoch+1}/{epochs} completed. MSE Loss: {loss:.4f}")
            self.backward(X, y)

    def predict(self, X):
        A2 = self.forward(X)
        return (A2 > 0.5).astype(int)
    
if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n = 100
    d = 10
    h = 30
    o = 1

    X = np.random.randn(n, d)  # 100 samples, 3 features
    W = np.random.randn(d, 1)
    y = (np.dot(X, W) > 0).astype(int) 
    # y = (np.sum(X, axis=1, keepdims=True) > 1.5).astype(int)  # Binary target

    nn = NeuralNetwork(input_size=d, hidden_size=h, output_size=1, learning_rate=0.1)
    nn.train(X, y, epochs=8000)

    predictions = nn.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Training accuracy: {accuracy * 100:.2f}%")