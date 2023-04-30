import numpy as np
from scipy.linalg import lstsq

A = np.array([
    [0.16, 0.10],
    [0.17, 0.11],
    [2.02, 1.29]
])

b1 = np.array([0.26, 0.28, 3.31]).reshape(-1, 1)
b2 = np.array([0.27, 0.25, 3.33]).reshape(-1, 1)

def least_sqaures(A, b):
    return np.linalg.inv(A.T @ A) @ A.T @ b

x1 = least_sqaures(A, b1)
x2 = least_sqaures(A, b2)

np.set_printoptions(precision=20)
print(f"result (a) : {x1})")
print(f"result (b) : {x2})")

difference = np.abs(x1 - x2)
print(f"difference : {difference}")