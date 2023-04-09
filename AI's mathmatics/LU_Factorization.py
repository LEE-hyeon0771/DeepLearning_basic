import numpy as np

def LU_Factorization(A):
    n = len(A)
    L = np.eye(n, dtype=np.float64)
    U = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        U[i][i] = A[i][i]
        for k in range(i+1, n):
            L[k][i] = A[k][i] / U[i][i]
            U[i][k] = A[i][k]
        for j in range(i+1, n):
            for k in range(i+1, n):
                A[j][k] = A[j][k] - L[j][i] * U[i][k]

    return L, U


A = np.array([[1,2,1], [2,3,1], [3,-2,-3]], dtype=np.float64)
b = np.array([[3], [4], [-1]], dtype=np.float64)

L, U = LU_Factorization(A)
print(f"LU분해를 이용한 L : \n{L}")
print(f"LU분해를 이용한 U : \n{U}")