import numpy as np
from scipy.linalg import lstsq

A = np.array([
    [0.16, 0.10],
    [0.17, 0.11],
    [2.02, 1.29]
])

b1 = np.array([0.26, 0.28, 3.31]).reshape(-1, 1)
b2 = np.array([0.27, 0.25, 3.33]).reshape(-1, 1)

x1, _, _, _ = lstsq(A, b1)
x2, _, _, _ = lstsq(A, b2)

np.set_printoptions(precision=20)
print(f"result (a) : {x1})")
print(f"result (b) : {x2})")

difference = np.abs(x1 - x2)
print(f"difference : {difference}")