#  Write a python program to show Back Propagation Network for XOR function with Binary Input and Output.

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(1)
input_size = 2
hidden_size = 2
output_size = 1

W1 = 2 * np.random.random((input_size, hidden_size)) - 1
W2 = 2 * np.random.random((hidden_size, output_size)) - 1

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    hidden_input = np.dot(X, W1)
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2)
    final_output = sigmoid(final_input)

    error = y - final_output

    d_final = error * sigmoid_derivative(final_output)
    error_hidden = d_final.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    W2 += learning_rate * hidden_output.T.dot(d_final)
    W1 += learning_rate * X.T.dot(d_hidden)

    if (epoch + 1) % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

print("\nFinal outputs after training:")
print(final_output)
