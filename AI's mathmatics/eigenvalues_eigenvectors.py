import numpy as np

# (a) 행렬 정의
A1 = np.array([[1, 4], [4, 2]])

# (b) 행렬 정의
A2 = np.array([[-1, -1, 1], [-1, 4, 2], [1, 2, 4]])

# 아이겐밸류와 아이겐벡터 계산
eigenvalues1, eigenvectors1 = np.linalg.eigh(A1)
eigenvalues2, eigenvectors2 = np.linalg.eigh(A2)

# 결과 출력
print("For matrix A1:")
print("Eigenvalues are:\n", eigenvalues1)
print("Eigenvectors are:\n", eigenvectors1)

print("For matrix A2:")
print("Eigenvalues are:\n", eigenvalues2)
print("Eigenvectors are:\n", eigenvectors2)

normalized_eigenvectors1 = eigenvectors1 / np.linalg.norm(eigenvectors1, axis=0)

# (b) 행렬의 아이겐벡터 정규화
normalized_eigenvectors2 = eigenvectors2 / np.linalg.norm(eigenvectors2, axis=0)

# 결과 출력
print("For matrix A1:")
print("Normalized eigenvectors are:\n", normalized_eigenvectors1)

print("For matrix A2:")
print("Normalized eigenvectors are:\n", normalized_eigenvectors2)