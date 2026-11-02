import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(0)
W1 = np.random.uniform(-1, 1, (2, 2))
W2 = np.random.uniform(-1, 1, (2, 1))
b1 = np.zeros((1, 2))
b2 = np.zeros((1, 1))
lr = 0.5

def forward_propagation(X):
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)
    return hidden_input, hidden_output, final_input, final_output

def backward_propagation(X, y, hidden_output, final_output):
    global W1, W2, b1, b2
    error = y - final_output
    d_output = error * sigmoid_derivative(final_output)
    hidden_error = d_output.dot(W2.T)
    d_hidden = hidden_error * sigmoid_derivative(hidden_output)
    W2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

for epoch in range(10000):
    hidden_input, hidden_output, final_input, final_output = forward_propagation(X)
    backward_propagation(X, y, hidden_output, final_output)
    if epoch % 1000 == 0:
        error = y - final_output
        print(f"Epoch {epoch} Error: {np.mean(np.abs(error))}")

print("Final predicted output:")
print(final_output)

binary_output = (final_output >= 0.5).astype(int)
print("\nFinal Output After Thresholding:")
print(binary_output)
