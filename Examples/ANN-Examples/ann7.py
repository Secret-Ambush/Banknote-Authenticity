# Write a python Program for Bidirectional Associative Memory with two pairs of vectors. 

import numpy as np

X = np.array([[1, -1, 1], [-1, 1, -1]])
Y = np.array([[1, 1, -1], [-1, -1, 1]])

W = np.dot(X.T, Y)

def recall(input_vector, weight_matrix, is_forward=True):
    output = np.sign(np.dot(input_vector, weight_matrix))
    output[output == 0] = 1
    return output

test_X = np.array([[1, -1, 1]])
recalled_Y = recall(test_X, W, is_forward=True)
print("Recalled Y for given X:", recalled_Y)

test_Y = np.array([[1, 1, -1]])
recalled_X = recall(test_Y, W.T, is_forward=False)
print("Recalled X for given Y:", recalled_X)
