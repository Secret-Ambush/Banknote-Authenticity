#  Write a python program in python program for creating a Back Propagation Feed-forward neural network 

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.5

np.random.seed(42)
W1 = 2 * np.random.random((input_size, hidden_size)) - 1
W2 = 2 * np.random.random((hidden_size, output_size)) - 1
b1 = np.zeros((1, hidden_size))
b2 = np.zeros((1, output_size))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

epochs = 10000
for epoch in range(epochs):
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    error = y - final_output
    d_output = error * sigmoid_derivative(final_output)

    hidden_error = d_output.dot(W2.T)
    d_hidden = hidden_error * sigmoid_derivative(hidden_output)

    W2 += learning_rate * hidden_output.T.dot(d_output)
    W1 += learning_rate * X.T.dot(d_hidden)
    b2 += learning_rate * np.sum(d_output, axis=0, keepdims=True)
    b1 += learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}: Loss = {loss:.5f}")

print("\nFinal Output After Training:")
print(final_output)

binary_output = (final_output >= 0.5).astype(int)
print("\nFinal Output After Thresholding:")
print(binary_output)

