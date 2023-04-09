import numpy as np
from numpy import linalg as LA

x = np.array([2,3])
y = np.array([3,6])

proj_x_y= (np.dot(x,y) * x) / np.dot(x,x)
cos_theta = np.dot(x,y) / (LA.norm(x) * LA.norm(y))
theta = np.arccos(cos_theta)

print(f"(a) : {x+y}")
print(f"(b) : {np.dot(x,y)}")
print(f"(c) : {proj_x_y}")
print(f"(d) : {theta}")
