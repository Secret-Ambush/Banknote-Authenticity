# Write a Python program to plot a few activation functions that are being used in neural networks.
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def tanh(x):
  return np.tanh(x)

def relu(x):
  return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
  return np.where(x > 0, x, alpha * x)

sigmoid_values = sigmoid(x)
tanh_values = tanh(x)
relu_values = relu(x)
leaky_values = leaky_relu(x)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.plot(x, sigmoid_values, label='sigmoid')
plt.title('sigmoid activation function')
plt.grid()
plt.legend()

plt.subplot(2,2,2)
plt.plot(x, tanh_values, label='tanh')
plt.title('tanh activation function')
plt.grid()
plt.legend()

plt.subplot(2,2,3)
plt.plot(x, relu_values, label='relu')
plt.title('relu activation function')
plt.grid()
plt.legend()

plt.subplot(2,2,4)
plt.plot(x, leaky_values, label='leaky_relu')     
plt.title('Leaky_relu activation function')
plt.grid()
plt.legend()
