import numpy as np
from scipy.linalg import lstsq

A = np.array([
    [0.16, 0.10],
    [0.17, 0.11],
    [2.02, 1.29]
])

b = np.array([0.26, 0.28, 3.31]).reshape(-1, 1)


x, _, _, _ = lstsq(A, b)

np.set_printoptions(precision=20)
print(f"result (a) : {x})")