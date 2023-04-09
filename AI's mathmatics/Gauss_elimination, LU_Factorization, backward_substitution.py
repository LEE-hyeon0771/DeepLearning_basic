import numpy as np


def gauss_elimination(A, b):
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            if A[j][i] != 0:

                m = A[j][i] / A[i][i]
                A[j][i+1:] -= m * A[i][i+1:]
                A[j][i] = 0
                b[j] -= m * b[i]
    return A, b

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


def backward_substitution(U, b):

    n = len(b)
    x = np.zeros((n))
    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= U[i,j]*x[j]
        if U[i,i] == 0:
            continue
        x[i] /= U[i,i]
    return x


# 주어진 문제의 행렬 A와 벡터 b
A = np.array([[3, -7, -2, 2], [-3, 5, 1, 0], [6, -4, 0, -5], [-9, 5, -5, 12]], dtype=np.float64)
b = np.array([[-9], [5], [7], [11]], dtype=np.float64)

# gauss_elimination을 사용하여 U를 구함.
U, b = gauss_elimination(A, b)
print(f"gauss_elimination을 이용한 U : \n{U}")
print()
# backward_substitution으로 Ax = b를 풀이한 x.
x = backward_substitution(U, b)
print(f"x:\n{x}")
print()
# LU분해를 위해 새로운 A matrix 생성
A = np.array([[3, -7, -2, 2], [-3, 5, 1, 0], [6, -4, 0, -5], [-9, 5, -5, 12]], dtype=np.float64)
b = np.array([[-9], [5], [7], [11]], dtype=np.float64)

# LU 분해를 사용하여 L과 U를 정확히 구함.
Lower, Upper = LU_Factorization(A)
print(f"LU분해를 이용한 L : \n{Lower}")
print(f"LU분해를 이용한 U : \n{Upper}")
print(f"LU = A를 보여주기 위한 matrix A : \n{Lower@Upper}")
