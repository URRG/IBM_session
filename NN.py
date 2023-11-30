Dataset Used - Wine

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

wine = load_wine()
X, y = wine.data, wine.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse=False, categories='auto')
y = encoder.fit_transform(y.reshape(-1, 1))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

               self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output_prob = self.softmax(self.output)
        return self.output_prob

    def backward(self, X, y):
        
        error = self.output_prob - y
        d_output = error
        d_hidden = np.dot(d_output, self.weights_hidden_output.T) * (self.hidden_output * (1 - self.hidden_output))

        
        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_output.T, d_output)
        self.bias_output -= self.learning_rate * np.sum(d_output, axis=0, keepdims=True)
        self.weights_input_hidden -= self.learning_rate * np.dot(X.T, d_hidden)
        self.bias_hidden -= self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    def train(self, X, y, epochs):
        self.training_loss = []

        for epoch in range(epochs):
            output_prob = self.forward(X)
            loss = -np.sum(y * np.log(output_prob + 1e-15)) / len(X)
            self.training_loss.append(loss)
            self.backward(X, y)

        if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss}")



input_size = X_train.shape[1]
hidden_size = 8
output_size = y_train.shape[1]

nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(X_train, y_train, epochs=660)


y_pred_prob = nn.forward(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)


y_test_decoded = np.argmax(y_test, axis=1)


accuracy = accuracy_score(y_test_decoded, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


plt.plot(range(1, 661), nn.training_loss)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss Over Epochs')
plt.show()
