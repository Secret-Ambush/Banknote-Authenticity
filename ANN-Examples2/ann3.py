# With a suitable example demonstrate the perceptron learning law with its decision regions using
python. Give the output in graphical form.
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

x = np.random.rand(100, 2)  

y = np.array([1 if (x1 + x2) > 1 else -1 for x1, x2 in x])

weights = np.zeros(2)

bias = 0

learning_rate=0.1

max_epoch = 100

def train_perceptron(x, y, weights, bias, learning_rate, max_epoch):
    for epoch in range(max_epoch):
        errors = 0
        for i in range(len(x)):
            linear_output = np.dot(x[i], weights) + bias
            prediction = 1 if linear_output >= 0 else -1
            update = learning_rate * (y[i] - prediction)
            if update != 0:
                weights = weights + update * x[i]
                bias = bias + update
                errors += 1
        print(f"Epoch {epoch + 1}: number of errors = {errors}")
        if errors == 0:
            break
    return weights, bias   
  
weights, bias = train_perceptron(x, y, weights, bias, learning_rate, max_epoch)

def plot_decision_boundary(x, y, weights, bias):
    plt.figure(figsize=(8,5))
    plt.scatter(x[:,0], x[:,1], c=y, cmap='bwr', edgecolor='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    x_vals = np.array([x[:,0].min(), x[:,0].max()])
    y_vals = -(weights[0] * x_vals + bias) / weights[1]
    plt.plot(x_vals, y_vals, 'g--', label="Decision Boundary")
    plt.legend()
    plt.title("Perceptron Decision Boundary")
    plt.show() 
    
    
plot_decision_boundary(x, y, weights, bias)