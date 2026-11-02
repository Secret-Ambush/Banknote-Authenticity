""" Write a Python Program using Perceptron Neural Network to recognise even and odd numbers.
Given numbers are in ASCII form 0 to 9"""


import numpy as np

def ascii_to_binary(a):
    return np.array([int(bit) for bit in format(a, '08b')])

def train(X, y, lr=1.0, epochs=100):
    w = np.zeros(X.shape[1])
    b = 0.0
    for epoch in range(epochs):
        errors = 0
        for xi, t in zip(X, y):
            output = 1 if np.dot(w, xi) + b >= 0 else -1
            if output != t:
                w += lr * t * xi
                b += lr * t
                errors += 1
        if errors == 0:
            break
    return w, b

digits = list("0123456789")
X = np.array([ascii_to_binary(ord(d)) for d in digits])
y = np.array([1 if int(d) % 2 == 0 else -1 for d in digits])

w, b = train(X, y)

for d, xi in zip(digits, X):
    out = 1 if np.dot(w, xi) + b >= 0 else -1
    print(f"Digit: {d}, Classified as: {'Even' if out == 1 else 'Odd'}")
