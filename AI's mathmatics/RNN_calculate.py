import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def tanh(x):
    return np.tanh(x)

# Parameters
U = np.array([[0.1, 0.1], [0.0, 0.0], [0.0, -0.1]])
W = np.array([[0.1, 0.1, 0.0], [0.0, 0.0, 0.0], [0.2, -0.1, -0.1]])
V = np.array([[0.0, 0.1, 0.0], [-0.2, 0.0, 0.0]])
b = np.array([0.0, 0.0, 0.2])
c = np.array([0.2, 0.1])
x = np.array([[0.0, 1.0], [0.0, 0.1], [0.1, -0.2], [0.5, 0.0]])


h = np.array([0.0, 0.0, 0.0])

# t=1
a1 = np.dot(W, h) + np.dot(U, x[0]) + b
h1 = tanh(a1)
y1 = softmax(np.dot(V, h1) + c)

# t=2
a2 = np.dot(W, h1) + np.dot(U, x[1]) + b
h2 = tanh(a2)
y2 = softmax(np.dot(V, h2) + c)

print("Output at t=1:", y1)
print("Output at t=2:", y2)