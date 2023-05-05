import numpy as np

A = np.array([[1, 2],
              [2, 1]])
x0 = np.array([1, 0]).reshape(-1, 1)

def normalized_power_iteration(A, x0, num_iterations=100):
    x = x0
    for _ in range(num_iterations):
        y = np.dot(A, x)
        x = y / np.linalg.norm(y, 2)
    return x

result = normalized_power_iteration(A, x0)
print(result)